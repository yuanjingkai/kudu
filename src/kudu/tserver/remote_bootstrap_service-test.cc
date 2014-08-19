// Copyright (c) 2014, Cloudera, inc.
#include "kudu/tserver/tablet_server-test-base.h"

#include <boost/foreach.hpp>
#include <gflags/gflags.h>
#include <limits>

#include "kudu/consensus/log.h"
#include "kudu/consensus/log_util.h"
#include "kudu/consensus/opid_anchor_registry.h"
#include "kudu/rpc/rpc_header.pb.h"
#include "kudu/rpc/transfer.h"
#include "kudu/tserver/remote_bootstrap.pb.h"
#include "kudu/server/metadata.pb.h"
#include "kudu/tserver/tserver_service.pb.h"
#include "kudu/tserver/tserver_service.proxy.h"
#include "kudu/util/crc.h"
#include "kudu/util/env_util.h"
#include "kudu/util/monotime.h"
#include "kudu/util/stopwatch.h"
#include "kudu/util/test_util.h"

#define ASSERT_REMOTE_ERROR(status, err, code, str) \
    ASSERT_NO_FATAL_FAILURE(AssertRemoteError(status, err, code, str))

DECLARE_uint64(remote_bootstrap_idle_timeout_ms);
DECLARE_uint64(remote_bootstrap_timeout_poll_period_ms);

namespace kudu {
namespace tserver {

using env_util::ReadFully;
using log::MaximumOpId;
using log::MinimumOpId;
using log::OpIdEquals;
using log::ReadableLogSegment;
using log::ReadableLogSegmentMap;
using rpc::ErrorStatusPB;
using rpc::RpcController;

static const int kNumRolls = 2;

class RemoteBootstrapServiceTest : public TabletServerTest {
 public:
  RemoteBootstrapServiceTest()
    : flag_saver_(new google::FlagSaver()) {
  }

  virtual void SetUp() OVERRIDE {
    // Poll for session expiration every 10 ms for the session timeout test.
    FLAGS_remote_bootstrap_timeout_poll_period_ms = 10;
    TabletServerTest::SetUp();
    // Prevent logs from being deleted out from under us until / unless we want
    // to test that we are anchoring correctly. Since GenerateTestData() does a
    // Flush(), Log GC is allowed to eat the logs before we get around to
    // starting a remote bootstrap session.
    tablet_peer_->tablet()->opid_anchor_registry()->Register(MinimumOpId(), CURRENT_TEST_NAME(),
                                                             &anchor_);
    GenerateTestData();
  }

  virtual void TearDown() OVERRIDE {
    ASSERT_OK(tablet_peer_->tablet()->opid_anchor_registry()->Unregister(&anchor_));
    TabletServerTest::TearDown();
  }

 protected:
  void GenerateTestData() {
    const int kIncr = 50;
    LOG_TIMING(INFO, "Loading test data") {
      for (int row_id = 0; row_id < kNumRolls * kIncr; row_id += kIncr) {
        InsertTestRowsRemote(0, row_id, kIncr);
        CHECK_OK(tablet_peer_->tablet()->Flush());
        CHECK_OK(tablet_peer_->log()->AllocateSegmentAndRollOver());
      }
    }
  }

  const std::string GetLocalUUID() const {
    // FIXME: fs_manager()->uuid() fails. Problem?
    //return mini_server_->fs_manager()->uuid();
    return CURRENT_TEST_NAME();
  }

  const std::string& GetTabletId() const {
    return tablet_peer_->tablet()->tablet_id();
  }

  Status DoBeginRemoteBootstrapSession(const string& tablet_id,
                                       const string& requestor_uuid,
                                       BeginRemoteBootstrapSessionResponsePB* resp,
                                       RpcController* controller) {
    controller->set_timeout(MonoDelta::FromSeconds(1.0));
    BeginRemoteBootstrapSessionRequestPB req;
    req.set_tablet_id(tablet_id);
    req.set_requestor_uuid(requestor_uuid);
    return UnwindRemoteError(proxy_->BeginRemoteBootstrapSession(req, resp, controller),
                             controller);
  }

  Status DoBeginValidRemoteBootstrapSession(string* session_id,
                                            metadata::TabletSuperBlockPB* superblock = NULL,
                                            uint64_t* idle_timeout_millis = NULL,
                                            vector<consensus::OpId>* first_op_ids = NULL) {
    BeginRemoteBootstrapSessionResponsePB resp;
    RpcController controller;
    RETURN_NOT_OK(DoBeginRemoteBootstrapSession(GetTabletId(), GetLocalUUID(), &resp, &controller));
    *session_id = resp.session_id();
    if (superblock) {
      *superblock = resp.superblock();
    }
    if (idle_timeout_millis) {
      *idle_timeout_millis = resp.session_idle_timeout_millis();
    }
    if (first_op_ids) {
      for (int i = 0; i < resp.first_op_ids_size(); i++) {
        first_op_ids->push_back(resp.first_op_ids(i));
      }
    }
    return Status::OK();
  }

  Status DoCheckSessionActive(const string& session_id,
                              CheckRemoteBootstrapSessionActiveResponsePB* resp,
                              RpcController* controller) {
    controller->set_timeout(MonoDelta::FromSeconds(1.0));
    CheckRemoteBootstrapSessionActiveRequestPB req;
    req.set_session_id(session_id);
    return UnwindRemoteError(proxy_->CheckSessionActive(req, resp, controller), controller);
  }

  Status DoFetchData(const string& session_id, const DataIdPB& data_id,
                     uint64_t* offset, int64_t* max_length,
                     FetchDataResponsePB* resp,
                     RpcController* controller) {
    controller->set_timeout(MonoDelta::FromSeconds(1.0));
    FetchDataRequestPB req;
    req.set_session_id(session_id);
    req.mutable_data_id()->CopyFrom(data_id);
    if (offset) {
      req.set_offset(*offset);
    }
    if (max_length) {
      req.set_max_length(*max_length);
    }
    return UnwindRemoteError(proxy_->FetchData(req, resp, controller), controller);
  }

  Status DoEndRemoteBootstrapSession(const string& session_id, bool is_success,
                                     const Status* error_msg,
                                     EndRemoteBootstrapSessionResponsePB* resp,
                                     RpcController* controller) {
    controller->set_timeout(MonoDelta::FromSeconds(1.0));
    EndRemoteBootstrapSessionRequestPB req;
    req.set_session_id(session_id);
    req.set_is_success(is_success);
    if (error_msg) {
      StatusToPB(*error_msg, req.mutable_error());
    }
    return UnwindRemoteError(proxy_->EndRemoteBootstrapSession(req, resp, controller), controller);
  }

  // Decode the remote error into a Status object.
  Status ExtractRemoteError(const ErrorStatusPB* remote_error) {
    const RemoteBootstrapErrorPB& error =
        remote_error->GetExtension(RemoteBootstrapErrorPB::remote_bootstrap_error_ext);
    return StatusFromPB(error.status());
  }

  // Enhance a RemoteError Status message with additional details from the remote.
  Status UnwindRemoteError(Status status, const RpcController* controller) {
    if (!status.IsRemoteError()) {
      return status;
    }
    Status remote_error = ExtractRemoteError(controller->error_response());
    return status.CloneAndPrepend(remote_error.ToString());
  }

  void AssertRemoteError(Status status, const ErrorStatusPB* remote_error,
                         const RemoteBootstrapErrorPB::Code app_code,
                         const string& status_code_string) {
    ASSERT_TRUE(status.IsRemoteError()) << "Unexpected status code: " << status.ToString()
                                        << ", app code: "
                                        << RemoteBootstrapErrorPB::Code_Name(app_code)
                                        << ", status code string: " << status_code_string;
    const Status app_status = ExtractRemoteError(remote_error);
    const RemoteBootstrapErrorPB& error =
        remote_error->GetExtension(RemoteBootstrapErrorPB::remote_bootstrap_error_ext);
    ASSERT_EQ(app_code, error.code()) << error.ShortDebugString();
    ASSERT_EQ(status_code_string, app_status.CodeAsString()) << app_status.ToString();
    LOG(INFO) << app_status.ToString();
  }

  static void AssertDataEqual(const uint8_t* local, int64_t size, const DataChunkPB& remote) {
    ASSERT_EQ(size, remote.data().size());
    ASSERT_EQ(0, ::memcmp(local, remote.data().data(), size));
    uint32_t crc32 = crc::Crc32c(local, size);
    ASSERT_EQ(crc32, remote.crc32());
  }

  // Read a block file from the file system fully into memory and return a
  // Slice pointing to it.
  Slice ReadLocalBlockFile(const BlockId& block_id, faststring* scratch) {
    shared_ptr<RandomAccessFile> block_file;
    CHECK_OK(mini_server_->fs_manager()->OpenBlock(block_id, &block_file));

    uint64_t size = 0;
    CHECK_OK(block_file->Size(&size));
    scratch->resize(size);
    Slice slice;
    CHECK_OK(ReadFully(block_file.get(), 0, size, &slice, scratch->data()));

    // Since the mmap will go away on return, copy the data into scratch.
    if (slice.data() != scratch->data()) {
      memcpy(scratch->data(), slice.data(), slice.size());
      slice = Slice(scratch->data(), slice.size());
    }
    return slice;
  }

  // Grab the first column block we find in the SuperBlock.
  static BlockId FirstColumnBlockId(const metadata::TabletSuperBlockPB& superblock) {
    const metadata::RowSetDataPB& rowset = superblock.rowsets(0);
    const metadata::ColumnDataPB& column = rowset.columns(0);
    const BlockIdPB& block_id_pb = column.block();
    return BlockId::FromPB(block_id_pb);
  }

  // Return BlockId in format suitable for a FetchData() call.
  static DataIdPB AsDataTypeId(const BlockId& block_id) {
    DataIdPB data_id;
    data_id.set_type(DataIdPB::BLOCK);
    block_id.CopyToPB(data_id.mutable_block_id());
    return data_id;
  }

  log::OpIdAnchor anchor_;
  gscoped_ptr<google::FlagSaver> flag_saver_;
};

// Test beginning and ending a remote bootstrap session.
TEST_F(RemoteBootstrapServiceTest, TestSimpleBeginEndSession) {
  string session_id;
  metadata::TabletSuperBlockPB superblock;
  uint64_t idle_timeout_millis;
  vector<consensus::OpId> first_op_ids;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id,
                                                      &superblock,
                                                      &idle_timeout_millis,
                                                      &first_op_ids));
  // Basic validation of returned params.
  ASSERT_FALSE(session_id.empty());
  ASSERT_EQ(FLAGS_remote_bootstrap_idle_timeout_ms, idle_timeout_millis);
  ASSERT_TRUE(superblock.IsInitialized());
  ASSERT_EQ(kNumRolls, first_op_ids.size());

  EndRemoteBootstrapSessionResponsePB resp;
  RpcController controller;
  ASSERT_OK(DoEndRemoteBootstrapSession(session_id, true, NULL, &resp, &controller));
}

// Test starting two sessions. The current implementation will silently only create one.
TEST_F(RemoteBootstrapServiceTest, TestBeginTwice) {
  // Second time through should silently succeed.
  for (int i = 0; i < 2; i++) {
    string session_id;
    ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id));
    ASSERT_FALSE(session_id.empty());
  }
}

// Test bad session id error condition.
TEST_F(RemoteBootstrapServiceTest, TestInvalidSessionId) {
  vector<string> bad_session_ids;
  bad_session_ids.push_back("hodor");
  bad_session_ids.push_back(GetLocalUUID());

  // Fetch a block for a non-existent session.
  BOOST_FOREACH(const string& session_id, bad_session_ids) {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.set_type(DataIdPB::BLOCK);
    data_id.mutable_block_id()->set_id("snarf");
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(), RemoteBootstrapErrorPB::NO_SESSION,
                        Status::NotFound("").CodeAsString());
  }

  // End a non-existent session.
  BOOST_FOREACH(const string& session_id, bad_session_ids) {
    EndRemoteBootstrapSessionResponsePB resp;
    RpcController controller;
    Status status = DoEndRemoteBootstrapSession(session_id, true, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(), RemoteBootstrapErrorPB::NO_SESSION,
                        Status::NotFound("").CodeAsString());
  }
}

// Test bad tablet id error condition.
TEST_F(RemoteBootstrapServiceTest, TestInvalidTabletId) {
  BeginRemoteBootstrapSessionResponsePB resp;
  RpcController controller;
  Status status =
      DoBeginRemoteBootstrapSession("some-unknown-tablet", GetLocalUUID(), &resp, &controller);
  ASSERT_REMOTE_ERROR(status, controller.error_response(), RemoteBootstrapErrorPB::TABLET_NOT_FOUND,
                      Status::NotFound("").CodeAsString());
}

// Test DataIdPB validation.
TEST_F(RemoteBootstrapServiceTest, TestInvalidBlockOrOpId) {
  string session_id;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id));

  // Too-short BlockId name.
  {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.set_type(DataIdPB::BLOCK);
    data_id.mutable_block_id()->set_id("gurgle");
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(),
                        RemoteBootstrapErrorPB::INVALID_REMOTE_BOOTSTRAP_REQUEST,
                        Status::InvalidArgument("").CodeAsString());
  }

  // Invalid BlockId.
  {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.set_type(DataIdPB::BLOCK);
    data_id.mutable_block_id()->set_id("8charnam");
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(),
                        RemoteBootstrapErrorPB::BLOCK_NOT_FOUND,
                        Status::NotFound("").CodeAsString());
  }

  // Invalid OpId for log fetch.
  {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.set_type(DataIdPB::LOG_SEGMENT);
    data_id.mutable_first_op_id()->CopyFrom(MaximumOpId());
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(),
                        RemoteBootstrapErrorPB::WAL_SEGMENT_NOT_FOUND,
                        Status::NotFound("").CodeAsString());
  }

  // Empty data type with BlockId.
  // The RPC system will not let us send the required type field.
  {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.mutable_block_id()->set_id("8charman");
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_TRUE(status.IsInvalidArgument());
  }

  // Empty data type id (no BlockId, no OpId);
  {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.set_type(DataIdPB::LOG_SEGMENT);
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(),
                        RemoteBootstrapErrorPB::INVALID_REMOTE_BOOTSTRAP_REQUEST,
                        Status::InvalidArgument("").CodeAsString());
  }

  // Both BlockId and OpId in the same "union" PB (illegal).
  {
    FetchDataResponsePB resp;
    RpcController controller;
    DataIdPB data_id;
    data_id.set_type(DataIdPB::BLOCK);
    data_id.mutable_block_id()->set_id("8charnam");
    data_id.mutable_first_op_id()->CopyFrom(MinimumOpId());
    Status status = DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller);
    ASSERT_REMOTE_ERROR(status, controller.error_response(),
                        RemoteBootstrapErrorPB::INVALID_REMOTE_BOOTSTRAP_REQUEST,
                        Status::InvalidArgument("").CodeAsString());
  }
}

// Test invalid file offset error condition.
TEST_F(RemoteBootstrapServiceTest, TestFetchInvalidBlockOffset) {
  string session_id;
  metadata::TabletSuperBlockPB superblock;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id, &superblock));

  FetchDataResponsePB resp;
  RpcController controller;
  // Impossible offset.
  uint64_t offset = std::numeric_limits<uint64_t>::max();
  Status status = DoFetchData(session_id, AsDataTypeId(FirstColumnBlockId(superblock)),
                              &offset, NULL, &resp, &controller);
  ASSERT_REMOTE_ERROR(status, controller.error_response(),
                      RemoteBootstrapErrorPB::INVALID_REMOTE_BOOTSTRAP_REQUEST,
                      Status::InvalidArgument("").CodeAsString());
}

// Test that we are able to fetch an entire block.
TEST_F(RemoteBootstrapServiceTest, TestFetchBlockAtOnce) {
  string session_id;
  metadata::TabletSuperBlockPB superblock;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id, &superblock));

  // Local.
  BlockId block_id = FirstColumnBlockId(superblock);
  faststring scratch;
  Slice local_data = ReadLocalBlockFile(block_id, &scratch);

  // Remote.
  FetchDataResponsePB resp;
  RpcController controller;
  ASSERT_OK(DoFetchData(session_id, AsDataTypeId(block_id), NULL, NULL, &resp, &controller));

  AssertDataEqual(local_data.data(), local_data.size(), resp.chunk());
}

// Test that we are able to incrementally fetch blocks.
TEST_F(RemoteBootstrapServiceTest, TestFetchBlockIncrementally) {
  string session_id;
  metadata::TabletSuperBlockPB superblock;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id, &superblock));

  BlockId block_id = FirstColumnBlockId(superblock);
  faststring scratch;
  Slice local_data = ReadLocalBlockFile(block_id, &scratch);

  // Grab the remote data in several chunks.
  int64_t block_size = local_data.size();
  int64_t max_chunk_size = block_size / 5;
  uint64_t offset = 0;
  while (offset < block_size) {
    FetchDataResponsePB resp;
    RpcController controller;
    ASSERT_OK(DoFetchData(session_id, AsDataTypeId(block_id),
                                 &offset, &max_chunk_size, &resp, &controller));
    int64_t returned_bytes = resp.chunk().data().size();
    ASSERT_LE(returned_bytes, max_chunk_size);
    AssertDataEqual(local_data.data() + offset, returned_bytes, resp.chunk());
    offset += returned_bytes;
  }
}

// Test that we are able to fetch log segments.
TEST_F(RemoteBootstrapServiceTest, TestFetchLog) {
  string session_id;
  metadata::TabletSuperBlockPB superblock;
  uint64_t idle_timeout_millis;
  vector<consensus::OpId> first_op_ids;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id,
                                                      &superblock,
                                                      &idle_timeout_millis,
                                                      &first_op_ids));
  ASSERT_EQ(kNumRolls, first_op_ids.size());
  const consensus::OpId& op_id = *first_op_ids.begin();

  // Fetch the remote data.
  FetchDataResponsePB resp;
  RpcController controller;
  DataIdPB data_id;
  data_id.set_type(DataIdPB::LOG_SEGMENT);
  data_id.mutable_first_op_id()->CopyFrom(op_id);
  ASSERT_OK(DoFetchData(session_id, data_id, NULL, NULL, &resp, &controller));

  // Fetch the local data.
  ReadableLogSegmentMap local_segments;
  tablet_peer_->log()->GetReadableLogSegments(&local_segments);
  ASSERT_TRUE(OpIdEquals(op_id, local_segments.begin()->first))
      << "Expected equal OpIds: " << op_id.ShortDebugString()
      << " and " << local_segments.begin()->first.ShortDebugString();
  const scoped_refptr<ReadableLogSegment>& segment = local_segments.begin()->second;
  faststring scratch;
  int64_t size = segment->file_size();
  scratch.resize(size);
  Slice slice;
  CHECK_OK(ReadFully(segment->readable_file().get(), 0, size, &slice, scratch.data()));

  AssertDataEqual(slice.data(), slice.size(), resp.chunk());
}

// Test that the remote bootstrap session timeout works properly.
TEST_F(RemoteBootstrapServiceTest, TestSessionTimeout) {
  // This flag should be seen by the service due to TSO.
  // We have also reduced the timeout polling frequency in SetUp().
  FLAGS_remote_bootstrap_idle_timeout_ms = 1; // Expire the session almost immediately.

  // Start session.
  string session_id;
  ASSERT_OK(DoBeginValidRemoteBootstrapSession(&session_id));

  MonoTime start_time = MonoTime::Now(MonoTime::FINE);
  CheckRemoteBootstrapSessionActiveResponsePB resp;

  do {
    RpcController controller;
    ASSERT_OK(DoCheckSessionActive(session_id, &resp, &controller));
    if (!resp.session_is_active()) {
      break;
    }
    usleep(1000); // 1 ms
  } while (MonoTime::Now(MonoTime::FINE).GetDeltaSince(start_time).ToSeconds() < 10);

  ASSERT_FALSE(resp.session_is_active()) << "Remote bootstrap session did not time out!";
}

} // namespace tserver
} // namespace kudu
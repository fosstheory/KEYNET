//! RPC interface for the transaction payment module.

use codec::Codec;
use jsonrpc_core::{Error as RpcError, ErrorCode, Result};
use jsonrpc_derive::rpc;
use online_profile::{LiveMachine, MachineId, RPCMachineInfo, RpcSysInfo, StakerInfo};
use online_profile_runtime_api::OpRpcApi as OpStorageRuntimeApi;
use sp_api::ProvideRuntimeApi;
use sp_blockchain::HeaderBackend;
use sp_rpc::number::NumberOrHex;
use sp_runtime::{
    generic::BlockId,
    traits::{Block as BlockT, MaybeDisplay},
};
use std::{convert::TryInto, sync::Arc};

#[rpc]
pub trait OpRpcApi<BlockHash, AccountId, ResponseType1, ResponseType2, ResponseType3, ResponseType4>
{
    #[rpc(name = "onlineProfile_getStakerNum")]
    fn get_total_staker_num(&self, at: Option<BlockHash>) -> Result<u64>;

    #[rpc(name = "onlineProfile_getOpInfo")]
    fn get_op_info(&self, at: Option<BlockHash>) -> Result<ResponseType1>;

    #[rpc(name = "onlineProfile_getStakerInfo")]
    fn get_staker_info(&self, account: AccountId, at: Option<BlockHash>) -> Result<ResponseType2>;

    #[rpc(name = "onlineProfile_getMachineList")]
    fn get_machine_list(&self, at: Option<BlockHash>) -> Result<ResponseType3>;

    #[rpc(name = "onlineProfile_getMachineInfo")]
    fn get_machine_info(
        &self,
        machine_id: MachineId,
        at: Option<BlockHash>,
    ) -> Result<ResponseType4>;
}

pub struct OpStorage<C, M> {
    client: Arc<C>,
    _marker: std::marker::PhantomData<M>,
}

impl<C, M> OpStorage<C, M> {
    pub fn new(client: Arc<C>) -> Self {
        Self { client, _marker: Default::default() }
    }
}

impl<C, Block, AccountId, Balance, BlockNumber>
    OpRpcApi<
        <Block as BlockT>::Hash,
        AccountId,
        RpcSysInfo<Balance>,
        StakerInfo<Balance>,
        LiveMachine,
        RPCMachineInfo<AccountId, BlockNumber, Balance>,
    > for OpStorage<C, Block>
where
    Block: BlockT,
    AccountId: Clone + std::fmt::Display + Codec + Ord,
    Balance: Codec + MaybeDisplay + Copy + TryInto<NumberOrHex>,
    BlockNumber: Clone + std::fmt::Display + Codec,
    C: Send + Sync + 'static,
    C: ProvideRuntimeApi<Block>,
    C: HeaderBackend<Block>,
    C::Api: OpStorageRuntimeApi<Block, AccountId, Balance, BlockNumber>,
{
    fn get_total_staker_num(&self, at: Option<<Block as BlockT>::Hash>) -> Result<u64> {
        let api = self.client.runtime_api();
        let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

        let runtime_api_result = api.get_total_staker_num(&at);
        runtime_api_result.map_err(|e| RpcError {
            code: ErrorCode::ServerError(9876),
            message: "Something wrong".into(),
            data: Some(format!("{:?}", e).into()),
        })
    }

    fn get_op_info(&self, at: Option<<Block as BlockT>::Hash>) -> Result<RpcSysInfo<Balance>> {
        let api = self.client.runtime_api();
        let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

        let runtime_api_result = api.get_op_info(&at);
        runtime_api_result.map_err(|e| RpcError {
            code: ErrorCode::ServerError(9876),
            message: "Something wrong".into(),
            data: Some(format!("{:?}", e).into()),
        })
    }

    fn get_staker_info(
        &self,
        account: AccountId,
        at: Option<<Block as BlockT>::Hash>,
    ) -> Result<StakerInfo<Balance>> {
        let api = self.client.runtime_api();
        let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

        let runtime_api_result = api.get_staker_info(&at, account);
        runtime_api_result.map_err(|e| RpcError {
            code: ErrorCode::ServerError(9876),
            message: "Something wrong".into(),
            data: Some(format!("{:?}", e).into()),
        })
    }

    fn get_machine_list(&self, at: Option<<Block as BlockT>::Hash>) -> Result<LiveMachine> {
        let api = self.client.runtime_api();
        let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

        let runtime_api_result = api.get_machine_list(&at);
        runtime_api_result.map_err(|e| RpcError {
            code: ErrorCode::ServerError(9876),
            message: "Something wrong".into(),
            data: Some(format!("{:?}", e).into()),
        })
    }

    fn get_machine_info(
        &self,
        machine_id: MachineId,
        at: Option<<Block as BlockT>::Hash>,
    ) -> Result<RPCMachineInfo<AccountId, BlockNumber, Balance>> {
        let api = self.client.runtime_api();
        let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

        let runtime_api_result = api.get_machine_info(&at, machine_id);
        runtime_api_result.map_err(|e| RpcError {
            code: ErrorCode::ServerError(9876),
            message: "Something wrong".into(),
            data: Some(format!("{:?}", e).into()),
        })
    }
}

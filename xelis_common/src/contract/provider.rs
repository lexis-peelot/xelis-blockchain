use xelis_vm::Tid;

use crate::{
    account::CiphertextCache,
    asset::AssetData,
    block::TopoHeight,
    crypto::{Hash, PublicKey}
};

use super::ContractStorage;

pub trait ContractProvider<'ty>: ContractStorage + Tid<'ty> {
    // Returns the balance of the contract
    fn get_contract_balance_for_asset(&self, contract: &Hash, asset: &Hash, topoheight: TopoHeight) -> Result<Option<(TopoHeight, u64)>, anyhow::Error>;

    // Get the account balance for asset
    fn get_account_balance_for_asset(&self, key: &PublicKey, asset: &Hash, topoheight: TopoHeight) -> Result<Option<(TopoHeight, CiphertextCache)>, anyhow::Error>;

    // Verify if an asset exists in the storage
    fn asset_exists(&self, asset: &Hash, topoheight: TopoHeight) -> Result<bool, anyhow::Error>;

    // Load the asset data from the storage
    fn load_asset_data(&self, asset: &Hash, topoheight: TopoHeight) -> Result<Option<(TopoHeight, AssetData)>, anyhow::Error>;

    // Load the asset supply
    fn load_asset_supply(&self, asset: &Hash, topoheight: TopoHeight) -> Result<Option<(TopoHeight, u64)>, anyhow::Error>;

    // Verify if the address is well registered
    fn account_exists(&self, key: &PublicKey, topoheight: TopoHeight) -> Result<bool, anyhow::Error>;
}
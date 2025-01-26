#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use alloy::{
    eips::BlockId,
    providers::{Provider, ProviderBuilder},
    rpc::types::Filter,
    primitives::address
};
use dotenv::dotenv;

use std::sync::Arc;
use std::path::Path;
use anyhow::Result;

use block_extractor_rs::{
    interfaces::IERC20,
    tokens::*,
    pools::*,
};



#[tokio::main]
async fn main() -> Result<()> {

    dotenv().ok();
    env_logger::init();

    let rpc_url = std::env::var("HTTP_URL")?;
    let provider = ProviderBuilder::new().on_http(rpc_url.parse()?);


    let pool_address_v3 = address!("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640");
    let pool_address_v2 = address!("0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc");
    let weth_address = address!("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");
    let usdc_address = address!("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48");

    let block_number = BlockId::from(provider.get_block_number().await.unwrap());
    let from_block_number = 8000000;
    let chunks = 50000;
    let parallel_chunks = 10;

    let (pools, i) = load_pools(
        provider.clone(),
        Path::new("./data/pools.csv"),
        from_block_number,
        chunks,
        parallel_chunks,
    ).await.unwrap();

    let parallel_tokens = 1;
    let tokens = load_tokens(
        provider.clone(),
        Path::new("./data/tokens.csv"),
        &pools,
        parallel_tokens
    ).await.unwrap();


    Ok(())

}

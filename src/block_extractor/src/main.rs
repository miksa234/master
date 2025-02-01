#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use alloy::{
    eips::BlockId,
    providers::{ProviderBuilder, WsConnect, Provider},
};
use arrow::compute::filter;
use dotenv::dotenv;
use log::info;
use std::path::Path;
use anyhow::{anyhow, Result};

use block_extractor_rs::{
    interfaces::*,
    tokens::*,
    pools::*,
    prices::*,
};


#[tokio::main]
async fn main() -> Result<()> {

    dotenv().ok();
    env_logger::init();

    let rpc_url = std::env::var("WS_URL")?;
    let ws = WsConnect::new(rpc_url);
    let provider = ProviderBuilder::new().on_ws(ws).await?;


    let block_number = BlockId::from(provider.get_block_number().await.unwrap());
    let from_block_number = 10000835;
    let chunks = 50000;

//    let (pools, pool_id) = load_pools(
//        provider.clone(),
//        Path::new("./data/pools.csv"),
//        from_block_number,
//        chunks,
//    ).await.unwrap();
//
//    let parallel_tokens = 1;
//    let tokens = load_tokens(
//        provider.clone(),
//        Path::new("./data/tokens.csv"),
//        &pools,
//        parallel_tokens,
//        pool_id,
//    ).await.unwrap();

    let filtered_pools = load_pools_from_file(
        Path::new("./data/filtered_pools.csv"),
    ).unwrap();

    let filtered_tokens = load_tokens_from_file(
        Path::new("./data/filtered_tokens.csv"),
    ).unwrap();

    info!("#fltered_pools {:?}", filtered_pools.len());
    info!("#filtered_tokens {:?}", filtered_tokens.len());

    //let p_from_block = 14763568;
    let p_from_block = 20015076;
    let p_to_block = 21715076;
    let block_gap = 36000; // approx 12 hours

    let prices = load_prices(
        provider.clone(),
        &filtered_pools,
        p_from_block,
        p_to_block,
        block_gap,
        Path::new("./data/prices.parquet")
    ).await.unwrap();



//    let prices = load_prices(
//        provider.clone()
//
//    );


    Ok(())

}

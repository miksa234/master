#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use alloy::{
    eips::BlockId,
    providers::{Provider, ProviderBuilder, WsConnect}, transports::http::reqwest::Url,
};
use arrow::compute::filter;
use dotenv::dotenv;
use log::info;
use std::path::Path;
use anyhow::{anyhow, Result};
use std::str::FromStr;
use std::sync::Arc;

use block_extractor_rs::{
    interfaces::*,
    tokens::*,
    pools::*,
    prices::*,
};

use std::fs::File;
use parquet::file::reader::{FileReader, SerializedFileReader};


#[tokio::main]
async fn main() -> Result<()> {

    dotenv().ok();
    env_logger::init();

//    let rpc_url = std::env::var("WSS_URL")?;
//    let ws = WsConnect::new(rpc_url);
//    let provider = ProviderBuilder::new().on_ws(ws).await?;

    let https_url = std::env::var("HTTPS_URL").unwrap();
    let provider = ProviderBuilder::new().on_builtin(
        https_url.as_str()
    ).await?;

//    let block_number = BlockId::from(provider.get_block_number().await.unwrap());
//    let from_block_number = 10000835;
//    let chunks = 50000;
//    let (pools, pool_id) = load_pools(
//        provider.clone(),
//        Path::new("../data/pools.csv"),
//        from_block_number,
//        chunks,
//    ).await.unwrap();
//
//    let parallel_tokens = 1;
//    let tokens = load_tokens(
//        provider.clone(),
//        Path::new("../data/tokens.csv"),
//        &pools,
//        parallel_tokens,
//        pool_id,
//    ).await.unwrap();
//
    let filtered_pools = load_pools_from_file(
        Path::new("../data/pools/pools_deg_5_liq_100_block_18_grad.csv"),
    ).unwrap();

    let tokens = load_tokens_from_file(
        Path::new("../data/tokens/tokens.csv"),
    ).unwrap();

    info!("#fltered_pools {:?}", filtered_pools.len());
    info!("#tokens {:?}", tokens.len());

    let p_to_block = provider.get_block_number().await?;
    let p_from_block = 18000000;
    let block_gap = 3600; // approx 12 hours

    let prices = load_prices(
        provider.clone(),
        &filtered_pools,
        p_from_block,
        p_to_block,
        block_gap,
        Path::new("../data/prices/prices_deg_5_liq_100_block_18_grad.parquet")
    ).await.unwrap();

    info!("Done len prices: {:?}", prices.len());

    Ok(())

}

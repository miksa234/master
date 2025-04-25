use alloy::{
    eips::BlockId,
    primitives::{Address, U256},
    providers::RootProvider,
    pubsub::PubSubFrontend, transports::BoxTransport,
};
use arrow::{
    array::ArrayRef, datatypes::{DataType, Field, Schema}, record_batch::RecordBatch
};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter}, basic::Compression, file::properties::{EnabledStatistics, WriterProperties}
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::BTreeMap,
    fs::{OpenOptions, File},
    path::Path,
    sync::Arc
};
use log::info;
use anyhow::{anyhow, Result};

use crate::{interfaces::*, pools::{Pool, Version}};

pub struct Price {
    pool: Address,
    block: u64,
    r_t0: Option<U256>,
    r_t1: Option<U256>,
    spx96: Option<U256>,
}


pub async fn load_prices(
    provider: RootProvider<BoxTransport>,
    pools: &BTreeMap<Address, Pool>,
    from_block: u64,
    to_block: u64,
    block_gap: u64,
    path : &Path,
) -> Result<Vec<Price>> {

    let mut prices = Vec::new();


    let mut blocks = Vec::new();
    blocks.push(from_block);
    let mut cur = from_block;
    loop {
        cur += block_gap;
        if cur > to_block {
            blocks.push(to_block);
            break
        }
        blocks.push(cur);
    }

    let pb = ProgressBar::new(blocks.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    pb.set_message(format!("From block {:?} - To Block {:?}", from_block, to_block));
    pb.inc(0);

    for block in blocks {
        'pool_loop: for (_, pool) in pools.into_iter() {
            match pool.version {
                Version::V2 => {
                    match get_v2_price(
                        provider.clone(),
                        block,
                        pool,
                    ).await {
                        Ok(price) => {
                            prices.push(
                                Price {
                                    pool: pool.address,
                                    block,
                                    r_t0: Some(price.0),
                                    r_t1: Some(price.1),
                                    spx96: None,
                                }
                            );
                        }
                        Err(e) => {
                            info!("Error getting price {:?}", e);
                            continue 'pool_loop;
                        }
                    };
                }
                Version::V3 => {
                    match get_v3_price(
                        provider.clone(),
                        block,
                        pool
                    ).await {
                        Ok(price) => {
                            prices.push(
                                Price {
                                    pool: pool.address,
                                    block,
                                    r_t0: Some(price.0),
                                    r_t1: Some(price.1),
                                    spx96: Some(price.2),
                                }
                            );
                        }
                        Err(e) => {
                            info!("Error getting price {:?}", e);
                            continue 'pool_loop;
                        }
                    };
                }
            }
        }
        pb.inc(1)
    }

    let file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path)
        .unwrap();

    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();

    let batch = create_record_batch(&prices).unwrap();
    println!("{:?}", batch.schema());

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    Ok(prices)
}

fn create_record_batch(
    prices: &Vec<Price>,
) -> Result<RecordBatch> {

    let pools = prices.iter()
        .map(|p| format!("{:?}", p.pool))
        .collect::<Vec<String>>();

    let blocks = prices.iter()
        .map(|p| p.block as i64)
        .collect::<Vec<i64>>();

    let r_t0s = prices.iter()
        .map(|p| match p.r_t0{
            Some(r) => Some(format!("{:?}", r)),
            None => None,
        }).collect::<Vec<Option<String>>>();

    let r_t1s = prices.iter()
        .map(|p| match p.r_t1{
            Some(r) => Some(format!("{:?}", r)),
            None => None,
        })
        .collect::<Vec<Option<String>>>();

    let spx96s = prices.iter()
        .map(|p| match p.spx96 {
            Some(r) => Some(format!("{:?}", r)),
            None => None,
        })
        .collect::<Vec<Option<String>>>();

    let batch = RecordBatch::try_from_iter(
        vec![
            ("pool_address", Arc::new(arrow::array::StringArray::from(pools)) as ArrayRef),
            ("block_number", Arc::new(arrow::array::Int64Array::from(blocks)) as ArrayRef),
            ("reserve_t0", Arc::new(arrow::array::StringArray::from(r_t0s)) as ArrayRef),
            ("reserve_t1", Arc::new(arrow::array::StringArray::from(r_t1s)) as ArrayRef),
            ("sqrt_price_x96", Arc::new(arrow::array::StringArray::from(spx96s)) as ArrayRef),
        ]
    ).unwrap();

    Ok(batch)
}

async fn get_v2_price(
    provider: RootProvider<BoxTransport>,
    block_number: u64,
    pool: &Pool,
) -> Result<(U256, U256)> {

    let block = BlockId::from(block_number);

    let token0_ierc20 = IERC20::new(pool.token0, &provider); // token1
    let token1_ierc20 = IERC20::new(pool.token1, &provider); // token1

    let balance_token0 = match token0_ierc20
        .balanceOf(pool.address)
        .block(block)
        .call()
        .await {
            Ok(r) => r.balance,
            Err(e) => { return Err(anyhow!("Error getting balance_token0 {:?}", e)); }
    };

    let balance_token1 = match token1_ierc20
        .balanceOf(pool.address)
        .block(block)
        .call()
        .await {
            Ok(r) => r.balance,
            Err(e) => { return Err(anyhow!("Error getting balance_token1 {:?}", e)); }
    };

    return Ok((balance_token0, balance_token1));
}

async fn get_v3_price(
    provider: RootProvider<BoxTransport>,
    block_number: u64,
    pool: &Pool,
) -> Result<(U256, U256, U256)> {

    let block = BlockId::from(block_number);

    let token0_ierc20 = IERC20::new(pool.token0, &provider); // token1
    let token1_ierc20 = IERC20::new(pool.token1, &provider); // token1
    let pool_int = IUniswapV3Pool::new(pool.address, &provider); // token1


    let balance_token0 = match token0_ierc20
        .balanceOf(pool.address)
        .block(block)
        .call()
        .await {
            Ok(r) => r.balance,
            Err(e) => { return Err(anyhow!("Error getting balance_token0 {:?}", e)); }
    };
    let balance_token1 = match token1_ierc20
        .balanceOf(pool.address)
        .block(block)
        .call()
        .await {
            Ok(r) => r.balance,
            Err(e) => { return Err(anyhow!("Error getting balance_token1 {:?}", e)); }
    };

    let sqrt_price_x96 = match pool_int
        .slot0()
        .block(block)
        .call()
        .await {
        Ok(r) => U256::from(r.sqrtPriceX96),
        Err(e) => { return Err(anyhow!("Error returning sqrt_price_x96 {:?}", e)); }
    };

    return Ok((balance_token0, balance_token1, sqrt_price_x96));
}

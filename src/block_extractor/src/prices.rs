use alloy::{
    eips::BlockId,
    primitives::{Address, U256},
    providers::RootProvider,
    pubsub::PubSubFrontend,
};
use arrow::{
    array::ArrayRef,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch
};
use parquet::{
    arrow::ArrowWriter,
    file::properties::WriterProperties
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    path::Path,
    sync::Arc
};
use log::info;
use anyhow::Result;

use crate::{interfaces::*, pools::{Pool, Version}};

pub struct Price {
    pool: Address,
    block: u64,
    price: U256
}

/*
    Price (marginal exchange rate) of the token is either
        token0/token1 or token1/token0
    It is determined by the decimals of the token
    the denominator needs to have less decimals then the
    numerator.
*/

pub async fn load_prices(
    provider: RootProvider<PubSubFrontend>,
    pools: &BTreeMap<Address, Pool>,
    from_block: u64,
    to_block: u64,
    block_gap: u64,
    path : &Path,
) -> Result<Vec<Price>> {

    let mut prices = Vec::new();

    let file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(path)
        .unwrap();

    let schema = Schema::new(Vec::from([
        Field::new("pool_address", DataType::Utf8, false),
        Field::new("block_number", DataType::Int64, false),
        Field::new("price", DataType::Utf8, false),
    ]));

    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props)).unwrap();

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
                        pool.address,
                        pool.token0,
                        18, // placeholder
                        pool.token1,
                        18 // placeholder
                    ).await {
                        Ok(price) => {
                            prices.push(
                                Price {
                                    pool: pool.address,
                                    block,
                                    price
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
                        pool.address,
                    ).await {
                        Ok(price) => {
                            prices.push(
                                Price {
                                    pool: pool.address,
                                    block,
                                    price
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

    let batch = create_record_batch(&prices, schema).unwrap();
    writer.write(&batch).unwrap();
    let _ = writer.close().unwrap();

    Ok(prices)
}

fn create_record_batch(
    prices: &Vec<Price>,
    schema: Schema,
) -> Result<RecordBatch> {

    let pools = prices.iter()
        .map(|p| format!("{:?}", p.pool))
        .collect::<Vec<String>>();

    let blocks = prices.iter()
        .map(|p| p.block as i64)
        .collect::<Vec<i64>>();

    let prices_vec = prices.iter()
        .map(|p| format!("{:?}", p.price))
        .collect::<Vec<String>>();

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        Vec::from([
            Arc::new(arrow::array::StringArray::from(pools)) as ArrayRef,
            Arc::new(arrow::array::Int64Array::from(blocks)) as ArrayRef,
            Arc::new(arrow::array::StringArray::from(prices_vec)) as ArrayRef
        ])
    )?;

    Ok(batch)
}

async fn get_v2_price(
    provider: RootProvider<PubSubFrontend>,
    block_number: u64,
    pool: Address,
    token0: Address,
    decimals_t0: u8,
    token1: Address,
    decimals_t1: u8,
) -> Result<U256> {

    let block = BlockId::from(block_number);

    let token0_ierc20 = IERC20::new(token0, &provider); // token1
    let token1_ierc20 = IERC20::new(token1, &provider); // token1

    let balance_token0 = token0_ierc20
        .balanceOf(pool)
        .block(block)
        .call()
        .await
        .unwrap()
        .balance;

    let balance_token1 = token1_ierc20
        .balanceOf(pool)
        .block(block)
        .call()
        .await
        .unwrap()
        .balance;

    let price;
    if decimals_t0 > decimals_t1 {
        // t0/t1
        price = balance_token0.checked_div(balance_token1).unwrap();
    } else {
        // t1/t0
        price = balance_token1.checked_div(balance_token0).unwrap();
    }

    return Ok(price);
}

async fn get_v3_price(
    provider: RootProvider<PubSubFrontend>,
    block_number: u64,
    pool: Address,
) -> Result<U256> {

    let block = BlockId::from(block_number);

    let pool_int = IUniswapV3Pool::new(pool, &provider); // token1
    let sqrt_price_x96 = pool_int
        .slot0()
        .block(block)
        .call()
        .await
        .unwrap()
        .sqrtPriceX96;

    // sqrt_price_x96 = sqrt(token0/token1) * 2**96
    // price = sqrt_price_x96**2 / 2**192
    let price = U256::from(sqrt_price_x96)
        .pow(U256::from(2))
        .checked_div(
            U256::from(2).pow(U256::from(192))
        ).unwrap();
    let price = U256::ZERO;

    return Ok(price);
}

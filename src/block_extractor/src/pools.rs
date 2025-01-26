#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_mut)]

use alloy::{
    primitives::{Address, FixedBytes},
    providers::{RootProvider, Provider},
    rpc::types::{BlockId, BlockTransactionsKind, Filter},
    sol_types::SolEvent,
    transports::http::{Client, Http}
};

use std::{
    collections::{BTreeMap, HashMap},
    fs::OpenOptions,
    path::Path,
    str::FromStr,
    sync::Arc
};

use indicatif::{ProgressBar, ProgressStyle};
use anyhow::{Result, anyhow};
use csv::StringRecord;
use log::info;

use crate::interfaces::{IUniswapV2Factory, IUniswapV3Factory};


#[derive(Debug, Clone)]
pub enum Version {
    V2,
    V3
}

#[derive(Debug, Clone)]
pub struct Pool {
    pub id: i64,
    pub address: Address,
    pub version: Version,
    pub token0: Address,
    pub token1: Address,
    pub fee: u32,
    pub block_number: u64,
    pub timestamp: u64,
    pub tickspacing: i32,
}

impl From<StringRecord> for Pool {
    fn from(record: StringRecord) -> Self {
        let version = match record.get(2).unwrap().parse().unwrap() {
            2 => Version::V2,
            _ => Version::V3
        };
        Self {
            id: record.get(0).unwrap().parse().unwrap(),
            address: Address::from_str(record.get(1).unwrap()).unwrap(),
            version,
            token0: Address::from_str(record.get(3).unwrap()).unwrap(),
            token1: Address::from_str(record.get(4).unwrap()).unwrap(),
            fee: record.get(5).unwrap().parse().unwrap(),
            block_number: record.get(6).unwrap().parse().unwrap(),
            timestamp: record.get(7).unwrap().parse().unwrap(),
            tickspacing: record.get(8).unwrap().parse().unwrap(),
        }
    }
}


impl Pool {
    pub fn cache_row(&self) -> (i64, String, i32, String, String, u32, u64, u64, i32) {
        (
            self.id,
            format!("{:?}", self.address),
            match self.version {
                Version::V2 => 2,
                _ => 3,
            },
            format!("{:?}", self.token0),
            format!("{:?}", self.token1),
            self.fee,
            self.block_number,
            self.timestamp,
            self.tickspacing,
        )
    }

    pub fn has_token(&self, token: Address) -> bool {
        self.token0 == token || self.token1 == token
    }
}

pub async fn load_pools(
    provider: RootProvider<Http<Client>>,
    path: &Path,
    from_block: u64,
    chunk: u64,
    parallel: u64
) -> Result<(BTreeMap<Address, Pool>, i64)> {

    info!("Loading Pools...");

    let mut pools = BTreeMap::new();
    let mut blocks = vec![];

    let file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(path)
        .unwrap();

    let mut writer = csv::Writer::from_writer(file);

    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        for row in reader.records() {
            let row = row.unwrap();
            let pool = Pool::from(row);
            blocks.push(pool.block_number);
            pools.insert(pool.address, pool);
        }
    } else {
        writer.write_record(&[
            "id",
            "address",
            "version",
            "token0",
            "token1",
            "fee",
            "block_number",
            "timestamp",
            "tickspacing",
        ])?;
    }

    let last_id = match pools.len() > 0{
        true => pools.last_key_value().unwrap().1.id,
        false => -1
    };

    let from_block = match last_id != -1 {
        true => {
            match blocks.iter().max() {
                Some(b) => *b,
                None => { return Err(anyhow!("load_pools could not find last processed block")); }
            }
        }
        false => from_block
    };


    let to_block = provider.get_block_number().await.unwrap();
//    let from_block = to_block;
    let mut processed_blocks = 0u64;
    let mut block_range: Vec<(u64, u64)> = vec![];

    info!("From block {:?} -> To block {:?}", from_block, to_block);

    loop {
        let start_idx = from_block + processed_blocks;
        let mut end_idx = start_idx + chunk - 1;
        if end_idx > to_block {
            end_idx = to_block;
            block_range.push((start_idx, end_idx));
            break;
        }
        block_range.push((start_idx, end_idx));
        processed_blocks += chunk;
    }

    let pb = ProgressBar::new(to_block-from_block);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut signatures = Vec::new();
    signatures.push(IUniswapV3Factory::PoolCreated::SIGNATURE_HASH);
    signatures.push(IUniswapV2Factory::PairCreated::SIGNATURE_HASH);

    let mut count = 0;
    let mut requests = vec![];
    for range in block_range {
        requests.push(
            tokio::task::spawn(
                get_pool_data(
                    provider.clone(),
                    range.0,
                    range.1,
                    signatures.clone()
                )
            )
        );
        count += 1;
        if count == parallel {
            count = 0;
            let results = futures::future::join_all(requests).await;
            for result in results {
                match result {
                    Ok(r) => match r {
                        Ok(pool_vec) => {
                            for p in pool_vec {
                                pools.insert(p.address, p);
                            }
                        }
                        _ => {}
                    }
                    _ => {}
                }
            }
            pb.inc(parallel*chunk)
        }
        requests = vec![];
    }

    let mut id = 0;
    let mut added = 0;

    for (_, pool) in pools.iter_mut() {
        if pool.id == -1 {
            id += 1;
            pool.id = id;
        }
        if (pool.id as i64) > last_id {
            writer.serialize(pool.cache_row())?;
            added += 1;
        }
    }
    writer.flush()?;

    Ok((pools, last_id))
}


async fn get_pool_data(
    provider: RootProvider<Http<Client>>,
    from_block: u64,
    to_block: u64,
    sig_hash: Vec<FixedBytes<32>>,
) -> Result<Vec<Pool>> {
    let mut pools = Vec::new();
    let mut timestamp_map: HashMap<u64, u64> = HashMap::new();

    let filter = Filter::new()
        .from_block(from_block)
        .to_block(to_block)
        .event_signature(sig_hash);

    let logs = provider.get_logs(&filter).await?;

    for log in logs {
        let (version, address, token0, token1, fee, tickspacing) = match log.topic0().unwrap() {
            &IUniswapV2Factory::PairCreated::SIGNATURE_HASH => {
                let event = IUniswapV2Factory::PairCreated::decode_log_data(
                    log.data(), true
                ).unwrap();
                let tickspacing: i32 = 0;
                let fee: u32 = 3000;
                (Version::V2, event.pair, event.token0, event.token1, fee, tickspacing)
            },
            &IUniswapV3Factory::PoolCreated::SIGNATURE_HASH => {
                let event = IUniswapV3Factory::PoolCreated::decode_log_data(
                    log.data(), true
                ).unwrap();
                (Version::V3, event.pool, event.token0, event.token1, event.fee.to::<u32>(), event.tickSpacing.as_i32())
            },
            _ => { continue; }
        };

        let block_number = log.block_number.unwrap();

        let timestamp = if !timestamp_map.contains_key(&block_number) {
            let block = provider.get_block(
                BlockId::from(block_number),
                BlockTransactionsKind::default()
            ).await.unwrap().unwrap();
            let timestamp = block.header.timestamp;
            timestamp
        } else {
            let timestamp  = *timestamp_map.get(&block_number).unwrap();
            timestamp
        };

        let pool_data = Pool {
            id: -1,
            address,
            version,
            token0,
            token1,
            fee,
            block_number,
            timestamp,
            tickspacing
        };

        pools.push(pool_data)
    }
    Ok(pools)
}











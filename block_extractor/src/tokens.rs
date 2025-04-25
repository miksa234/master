#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_mut)]

use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    path::Path,
    str::FromStr
};
use alloy::{
    primitives::Address,
    providers::RootProvider,
    transports::{http::{Client, Http}, BoxTransport},
};
use indicatif::{ProgressBar, ProgressStyle};
use anyhow::{anyhow, Result};
use csv::StringRecord;
use log::info;

use crate::{interfaces::IERC20, pools::Pool};


#[derive(Debug, Clone)]
pub struct Token {
    pub id: i64,
    pub address: Address,
    pub name: String,
    pub symbol: String,
    pub decimals: u8
}

impl From<StringRecord> for Token {
    fn from(record: StringRecord) -> Self {
        Self {
            id: record.get(0).unwrap().parse().unwrap(),
            address: Address::from_str(record.get(1).unwrap()).unwrap(),
            name: String::from(record.get(2).unwrap()),
            symbol: String::from(record.get(3).unwrap()),
            decimals: record.get(4).unwrap().parse().unwrap(),
        }
    }
}

impl Token {
    pub fn cache_row(&self) -> (i64, String, String, String, u8) {
        (
            self.id,
            format!("{:?}", self.address),
            self.name.clone(),
            self.symbol.clone(),
            self.decimals,
        )
    }
}

pub async fn load_tokens(
    provider: RootProvider<BoxTransport>,
    path: &Path,
    pools: &BTreeMap<Address, Pool>,
    parallel: u64,
    last_pool_id: i64,
) -> Result<BTreeMap<Address, Token>> {

    info!("Loading tokens...");

    let mut tokens = BTreeMap::new();

    let file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(path)
        .unwrap();

    let mut writer = csv::Writer::from_writer(file);

    let mut token_id = 0;
    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        for row in reader.records() {
            let row = row.unwrap();
            let token = Token::from(row);
            tokens.insert(token.address, token);
            token_id += 1;
        }
    } else {
        writer.write_record(&[
            "id",
            "address",
            "name",
            "symbol",
            "decimals",
        ])?;
    }

    let pb = ProgressBar::new(pools.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let new_token_id = token_id;

    let mut count = 0;
    let mut requests = Vec::new();

    for (_, pool) in pools.into_iter() {
        let pool_id = pool.id;
        if pool_id <= last_pool_id {
            continue;
        }
        let token0 = pool.token0;
        let token1 = pool.token1;
        for token in [token0, token1] {
            if !tokens.contains_key(&token) {
                requests.push(
                    tokio::task::spawn(
                        get_token_data(
                            provider.clone(),
                            token,
                        )
                    )
                );
                count += 1 ;
            }
            if count == parallel {
                let results = futures::future::join_all(requests).await;
                for result in results {
                    match result {
                        Ok(r) => match r {
                            Ok(t) => {
                                tokens.insert(
                                    t.address,
                                    Token {
                                        id: token_id,
                                        address: t.address,
                                        name: t.name,
                                        symbol: t.symbol,
                                        decimals: t.decimals
                                    }
                                );
                                token_id += 1
                            }
                            Err(e) => { info!("Something wrong 0 {:?}", e) }
                        }
                        Err(e) => { info!("Something wrong 1 {:?}", e) }
                    }
                }
                requests = Vec::new();
                count = 0;
                pb.inc(parallel);
            }
        }
    }

    let mut added = 0;
    for token in tokens.values().collect::<Vec<&Token>>().iter() {
        if token.id >= new_token_id {
            writer.serialize(token.cache_row())?;
            added += 1
        }
    }
    writer.flush()?;

    Ok(tokens)
}

async fn get_token_data(
    provider: RootProvider<BoxTransport>,
    token: Address,
) -> Result<Token> {

    let interface = IERC20::new(token, provider);

    let decimals = match interface.decimals().call().await {
        Ok(r) => r.decimals,
        Err(e) => { return Err(anyhow!("Decimals of token failed {:?}", e )) }
    };

    let name = match interface.name().call().await {
        Ok(r) => r.name,
        Err(e) => {
            info!("Name of token {:?} failed {:?}", token, e);
            String::from("PlaceHolderName")
        }
    };
    let symbol = match interface.symbol().call().await{
        Ok(r) => r.symbol,
        Err(e) => {
            info!("Symbol of token failed {:?}", e );
            String::from("PlaceHolderSymbol")
        }
    };

    Ok(Token {
        id: -1,
        address: token,
        name,
        symbol,
        decimals,
    })
}

pub fn load_tokens_from_file(
    path: &Path,
) -> Result<BTreeMap<Address, Token>> {
    let mut tokens = BTreeMap::new();

    if path.exists() {
        let mut reader = csv::Reader::from_path(path)?;
        for row in reader.records() {
            let row = row.unwrap();
            let token = Token::from(row);
            tokens.insert(token.address, token);
        }
    } else {
        return Err(anyhow!("File path does not exist"));
    }

    Ok(tokens)
}

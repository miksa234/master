use alloy::{
    eips::BlockId,
    primitives::{Address, U256},
    providers::Provider
};
use anyhow::Result;
use std::sync::Arc;

use crate::interfaces::*;

/*
    Price (marginal exchange rate) of the token is either
        token0/token1 or token1/token0
    It is determined by the decimals of the token
    the denominator needs to have less decimals then the
    numerator.
*/

async fn get_v2_price<P: Provider+ 'static>(
    provider: Arc<P>,
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

async fn get_v3_price<P: Provider+ 'static>(
    provider: Arc<P>,
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

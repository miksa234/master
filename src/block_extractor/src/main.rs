use alloy::{
    primitives::{address, U256},
    providers::{Provider, ProviderBuilder},
    eips::BlockId,
    sol
};

use anyhow::Result;

sol!(
    #[sol(rpc)]
    interface IUniswapV3Pool {
        function slot0() external view returns (
            uint160 sqrtPriceX96,
            int24 tick,
            uint16 observationIndex,
            uint16 observationCardinality,
            uint16 observationCardinalityNext,
            uint8 feeProtocol,
            bool unlocked
        );
    }
);

#[tokio::main]
async fn main() -> Result<()> {
    let rpc_url = "https://eth.merkle.io".parse()?;
    let provider = ProviderBuilder::new().on_http(rpc_url);

    let pool_address = address!("88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640");
    let block_number = BlockId::from(provider.get_block_number().await.unwrap());

    let pool = IUniswapV3Pool::new(pool_address, &provider);
    let sqrt_price_x96 = pool.slot0().block(block_number).call().await.unwrap().sqrtPriceX96;


    let sqrt_price = U256::from(sqrt_price_x96) / U256::from(2).pow(U256::from(96));
    let price = U256::from(10).pow(U256::from(18)) / sqrt_price.pow(U256::from(2));

    println!("sqrt_price_x96: {sqrt_price_x96:?}");
    println!("price: {price:?}");

    Ok(())
}

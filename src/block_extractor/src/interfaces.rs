use alloy::sol;

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

        function token0() external view returns (
            address adr
        );

        function token1() external view returns (
            address adr
        );

        function fee() external view returns (
            uint24 fee
        );
    }
);

sol!(
    #[sol(rpc)]
    interface IUniswapV2Pool {
        function getReserves() external view returns (
            uint112 reserve0,
            uint112 reserve1,
            uint32 blockTimestampLast
        );

        function token0() external view returns (
            address adr
        );

        function token1() external view returns (
            address adr
        );

        function fee() external view returns (
            uint24 fee
        );
    }
);

sol!(
    #[sol(rpc)]
    interface IERC20 {
        function balanceOf(address account) external view returns (
            uint256 balance
        );

        function decimals() external view returns (
            uint8 decimals
        );

        function name() external view returns (
            string name
        );

        function symbol() external view returns (
            string symbol
        );
    }
);

sol!(
    #[sol(rpc)]
    event PairCreated(address indexed token0, address indexed token1, address pair, uint);
    event PoolCreated(address indexed token0, address indexed token1, uint24 indexed fee, int24 tickSpacing, address pool);
);

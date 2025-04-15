This program extracts all UniswapV2 and UniswapV3 pools and their tokens.
Additionally it also extracts the last x blocks of price data of these pools.

To run the code you will need an eth node, best your own because its not
capped by rpc requests. Create an .env file and fill in the HTTP_URL of your
node e.g.:

~/.env
HTTP_URL="https://eth-mainnet.public.blastapi.io"

If you want logging add RUST_LOG=INFO to the .env file.

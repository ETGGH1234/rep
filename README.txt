# Enhanced Solana Copy Trading Bot - Setup Guide

## 🚀 Features

- **Priority Fee Controls**: Custom compute unit limits and pricing
- **Retry Logic**: Exponential backoff with configurable retry attempts
- **Slippage Management**: Per-trade slippage controls with safety validation
- **Validator Tips**: Optional MEV tips for better execution
- **Web Dashboard**: Real-time monitoring and configuration
- **Multi-DEX Support**: Jupiter, Orca, and Raydium (Jupiter fully implemented)

## 📋 Prerequisites

- Python 3.8+
- Solana wallet with some SOL for gas fees
- Target wallet address to copy trades from
- RPC endpoint (default: Solana mainnet)

## 🔧 Installation

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Wallet
Create a Solana wallet JSON file or use an existing one:
```bash
# If you don't have a wallet, create one with Solana CLI
solana-keygen new --outfile wallet.json

# Or export from Phantom/Solflare (save as wallet.json)
```

### 4. Create Configuration Directory
```bash
mkdir templates  # For web dashboard templates (auto-created)
```

## 🚀 Quick Start

### Option 1: Web Dashboard (Recommended)
```bash
python bot.py --web-dashboard --port 8000
```
Then open http://localhost:8000 in your browser.

### Option 2: Command Line Interface
```bash
python bot.py --target-wallet YOUR_TARGET_WALLET_ADDRESS --user-wallet-path wallet.json
```

## ⚙️ Configuration Options

### Web Dashboard Settings

Access the dashboard at `http://localhost:8000` and configure:

#### Bot Settings
- **Target Wallet**: Solana address to copy trades from
- **Position Size**: Percentage of original trade size (1-100%)
- **Slippage Tolerance**: Maximum acceptable slippage (0.1-5%)
- **Max Retries**: Number of retry attempts for failed transactions (1-10)

#### Priority Fee Settings
- **Compute Unit Limit**: Maximum compute units (50K-1.4M)
- **Compute Unit Price**: Price per compute unit in micro-lamports (0-100K)
- **Validator Tip**: Optional tip to validators in lamports (0-1M)

### Command Line Arguments
```bash
python bot.py [OPTIONS]

Options:
  --web-dashboard          Start web dashboard
  --port INTEGER          Dashboard port (default: 8000)
  --host TEXT             Dashboard host (default: 127.0.0.1)
  --config TEXT           Config file path (default: config.json)
  --target-wallet TEXT    Target wallet address
  --user-wallet-path TEXT Path to your wallet JSON
  --slippage FLOAT        Slippage tolerance %
  --compute-unit-limit INT Compute unit limit
  --compute-unit-price INT Compute unit price (μLamports)
  --tip-amount INT        Validator tip (lamports)
  --position-size FLOAT   Position size (0-1)
  --max-retries INT       Max retry attempts
```

## 🎛️ Advanced Configuration

### Priority Fee Optimization

**Compute Unit Limit**: 
- Default: 200,000
- Higher values = more complex transactions
- Too high = wasted fees
- Too low = transaction failure

**Compute Unit Price**:
- Default: 1,000 micro-lamports
- Higher = faster execution priority
- Calculate total fee: `(limit × price) / 1,000,000,000 SOL`

**Example**: 200K units × 5K micro-lamports = 0.001 SOL priority fee

### Retry Logic

The bot uses exponential backoff:
- Attempt 1: Immediate
- Attempt 2: 2 second delay  
- Attempt 3: 4 second delay
- Attempt 4: 8 second delay
- etc.

### Risk Management

- **5% Portfolio Limit**: Buy trades capped at 5% of total portfolio value
- **Slippage Validation**: Warning for slippage > 2%
- **Position Size Control**: Scale trades relative to original size

## 📊 Dashboard Features

### Real-time Monitoring
- Bot status (Running/Stopped)
- Trade statistics (Total, Successful, Failed, Retries)
- Live transaction log with WebSocket updates

### Interactive Controls
- Start/Stop bot with one click
- Adjust settings with sliders and inputs
- Real-time fee calculations
- Input validation with error messages

### Transaction Log
- Color-coded log levels (Success, Error, Warning, Info)
- Detailed transaction information
- Original vs copy trade signatures
- Error traces for debugging

## 🔒 Security Best Practices

### Wallet Security
```bash
# Set proper file permissions for wallet
chmod 600 wallet.json

# Use environment variables for sensitive data
export SOLANA_WALLET_PATH="/secure/path/wallet.json"
```

### Network Security
```bash
# Run dashboard on localhost only (default)
python bot.py --web-dashboard --host 127.0.0.1

# For remote access, use reverse proxy with SSL
# nginx, caddy, or cloudflare tunnel recommended
```

### Monitoring
```bash
# Check bot logs
tail -f copy_trading_bot.log

# Monitor system resources
htop  # CPU/Memory usage
```

## 🐛 Troubleshooting

### Common Issues

**"WebSocket subscription failed"**
```bash
# Check RPC endpoint
python -c "import aiohttp; print('Testing connection...')"
```

**"Failed to load wallet"**
```bash
# Verify wallet format
cat wallet.json | jq '.[0:10]'  # Should show array of numbers
```

**"Invalid Solana wallet address"**
```bash
# Validate address format (base58, 32-44 chars)
solana address --keypair wallet.json
```

**High failure rate**
- Increase compute unit limit
- Raise compute unit price
- Add validator tip
- Reduce position size

### Debug Mode
```bash
# Enable verbose logging
export PYTHONPATH=.
python -m logging.basicConfig level=DEBUG bot.py --web-dashboard
```

## 📈 Performance Optimization

### Fast Execution
- Use dedicated RPC endpoint (Helius, QuickNode)
- Increase compute unit price during high network congestion
- Set appropriate slippage (0.5-1% typical)

### Cost Optimization
- Monitor priority fee vs success rate
- Use dynamic compute unit limits
- Adjust position size based on portfolio

### Network Optimization
```bash
# Use GenesysGo RPC for better performance
--rpc-url https://ssc-dao.genesysgo.net/
```

## 🔧 Extending the Bot

### Adding New DEXs

1. Implement parser in `parse_[dex]_swap()` method
2. Add program ID to constants
3. Test with small trades first

### Custom Strategies

Modify `execute_copy_trade()` to implement:
- Dollar-cost averaging
- Take-profit/stop-loss
- Portfolio rebalancing
- Risk-based position sizing

### API Integration

Add webhooks or external API calls:
```python
# In execute_copy_trade method
async def notify_external_service(self, trade_info):
    # Send to Discord, Telegram, etc.
    pass
```

## 📞 Support

### Logs Location
- Application logs: `copy_trading_bot.log`
- Configuration: `config.json`
- Web dashboard: `http://localhost:8000`

### Performance Metrics
Access `/api/status` endpoint for programmatic monitoring:
```json
{
  "running": true,
  "stats": {
    "total_trades": 15,
    "successful_trades": 14,
    "failed_trades": 1,
    "total_retries": 3
  }
}
```

---

## ⚠️ Disclaimer

This bot is for educational purposes. Cryptocurrency trading involves substantial risk of loss. Always:
- Test with small amounts first
- Monitor performance closely  
- Have stop-loss mechanisms
- Don't invest more than you can afford to lose

## 📄 License

MIT License - Use at your own risk.1. Clone or Create Project Directory
```bash
mkdir solana-copy-bot
cd solana-copy-bot
```

###

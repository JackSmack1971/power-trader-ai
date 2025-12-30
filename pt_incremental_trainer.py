#!/usr/bin/env python3
"""
pt_incremental_trainer.py

Incremental trainer for walk-forward validation.
Prevents look-ahead bias by only training on data up to a specified timestamp.

Usage:
    python pt_incremental_trainer.py BTC --train-until 1704067200
    python pt_incremental_trainer.py ETH --train-until 1704067200 --output-dir ./BTC_models
"""

from kucoin.client import Market
market = Market(url='https://api.kucoin.com')
import time
import sys
import datetime
import traceback
import linecache
import json
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Incremental trainer for walk-forward validation')
parser.add_argument('coin', type=str, help='Coin symbol (e.g., BTC, ETH, DOGE)')
parser.add_argument('--train-until', type=int, required=True,
                    help='Unix timestamp - only use data before this time')
parser.add_argument('--output-dir', type=str, default='.',
                    help='Directory to save trained model files (default: current directory)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

args = parser.parse_args()

_arg_coin = args.coin.upper()
TRAIN_UNTIL_TS = args.train_until
OUTPUT_DIR = args.output_dir
VERBOSE = args.verbose

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("INCREMENTAL TRAINER - Walk-Forward Validation Mode")
print("=" * 60)
print(f"Coin: {_arg_coin}")
print(f"Training data limited to: {datetime.datetime.fromtimestamp(TRAIN_UNTIL_TS)}")
print(f"Unix timestamp: {TRAIN_UNTIL_TS}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE flag is set."""
    if VERBOSE:
        print(*args, **kwargs)

# Cache memory/weights in RAM (avoid re-reading and re-writing every loop)
_memory_cache = {}  # tf_choice -> dict(memory_list, weight_list, high_weight_list, low_weight_list, dirty)
_last_threshold_written = {}  # tf_choice -> float

def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_memory(tf_choice):
    """Load memories/weights for a timeframe once and keep them in RAM."""
    if tf_choice in _memory_cache:
        return _memory_cache[tf_choice]

    # Build paths with output directory
    mem_path = os.path.join(OUTPUT_DIR, f"memories_{tf_choice}.txt")
    weight_path = os.path.join(OUTPUT_DIR, f"memory_weights_{tf_choice}.txt")
    high_weight_path = os.path.join(OUTPUT_DIR, f"memory_weights_high_{tf_choice}.txt")
    low_weight_path = os.path.join(OUTPUT_DIR, f"memory_weights_low_{tf_choice}.txt")

    data = {
        "memory_list": [],
        "weight_list": [],
        "high_weight_list": [],
        "low_weight_list": [],
        "dirty": False,
    }
    try:
        data["memory_list"] = _read_text(mem_path).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split('~')
    except:
        data["memory_list"] = []
    try:
        data["weight_list"] = _read_text(weight_path).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
    except:
        data["weight_list"] = []
    try:
        data["high_weight_list"] = _read_text(high_weight_path).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
    except:
        data["high_weight_list"] = []
    try:
        data["low_weight_list"] = _read_text(low_weight_path).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
    except:
        data["low_weight_list"] = []
    _memory_cache[tf_choice] = data
    return data

def flush_memory(tf_choice, force=False):
    """Write memories/weights back to disk only when they changed (batch IO)."""
    data = _memory_cache.get(tf_choice)
    if not data:
        return
    if (not data.get("dirty")) and (not force):
        return

    # Build paths with output directory
    mem_path = os.path.join(OUTPUT_DIR, f"memories_{tf_choice}.txt")
    weight_path = os.path.join(OUTPUT_DIR, f"memory_weights_{tf_choice}.txt")
    high_weight_path = os.path.join(OUTPUT_DIR, f"memory_weights_high_{tf_choice}.txt")
    low_weight_path = os.path.join(OUTPUT_DIR, f"memory_weights_low_{tf_choice}.txt")

    try:
        with open(mem_path, "w+", encoding="utf-8") as f:
            f.write("~".join([x for x in data["memory_list"] if str(x).strip() != ""]))
    except:
        pass
    try:
        with open(weight_path, "w+", encoding="utf-8") as f:
            f.write(" ".join([str(x) for x in data["weight_list"] if str(x).strip() != ""]))
    except:
        pass
    try:
        with open(high_weight_path, "w+", encoding="utf-8") as f:
            f.write(" ".join([str(x) for x in data["high_weight_list"] if str(x).strip() != ""]))
    except:
        pass
    try:
        with open(low_weight_path, "w+", encoding="utf-8") as f:
            f.write(" ".join([str(x) for x in data["low_weight_list"] if str(x).strip() != ""]))
    except:
        pass
    data["dirty"] = False

def write_threshold_sometimes(tf_choice, perfect_threshold, loop_i, every=200):
    """Avoid writing neural_perfect_threshold_* every single loop."""
    last = _last_threshold_written.get(tf_choice)
    # write occasionally, or if it changed meaningfully
    if (loop_i % every != 0) and (last is not None) and (abs(perfect_threshold - last) < 0.05):
        return

    threshold_path = os.path.join(OUTPUT_DIR, f"neural_perfect_threshold_{tf_choice}.txt")
    try:
        with open(threshold_path, "w+", encoding="utf-8") as f:
            f.write(str(perfect_threshold))
        _last_threshold_written[tf_choice] = perfect_threshold
    except:
        pass

def should_stop_training(loop_i, every=50):
    """Check killer.txt less often (still responsive, way less IO)."""
    if loop_i % every != 0:
        return False
    try:
        killer_path = os.path.join(OUTPUT_DIR, "killer.txt")
        with open(killer_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip().lower() == "yes"
    except:
        return False

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj))

# Training configuration
coin_choice = _arg_coin + '-USDT'
tf_choices = ['1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week']
tf_minutes = [60, 120, 240, 480, 720, 1440, 10080]
number_of_candles = [2]
how_far_to_look_back = 100000

# Trainer status tracking
_trainer_started_at = int(time.time())
status_path = os.path.join(OUTPUT_DIR, "trainer_status.json")
try:
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coin": _arg_coin,
                "state": "TRAINING",
                "started_at": _trainer_started_at,
                "timestamp": _trainer_started_at,
                "train_until_ts": TRAIN_UNTIL_TS,
                "mode": "incremental"
            },
            f,
        )
except Exception:
    pass

the_big_index = 0

while the_big_index < len(tf_choices):
    tf_choice = tf_choices[the_big_index]
    print(f"\n{'='*60}")
    print(f"Training timeframe: {tf_choice}")
    print(f"{'='*60}")

    # Load existing memories for this timeframe
    _mem = load_memory(tf_choice)
    memory_list = _mem["memory_list"]
    weight_list = _mem["weight_list"]
    high_weight_list = _mem["high_weight_list"]
    low_weight_list = _mem["low_weight_list"]

    choice_index = tf_choices.index(tf_choice)
    timeframe = tf_choice
    timeframe_minutes = tf_minutes[choice_index]

    # CRITICAL: Set end time to TRAIN_UNTIL_TS to prevent look-ahead bias
    start_time = TRAIN_UNTIL_TS
    end_time = int(start_time - ((1500 * timeframe_minutes) * 60))

    print(f"Fetching historical data (limited to timestamp {TRAIN_UNTIL_TS})...")
    print(f"Date limit: {datetime.datetime.fromtimestamp(TRAIN_UNTIL_TS)}")

    history_list = []

    # Fetch historical data in chunks (KuCoin limit: 1500 candles per request)
    while True:
        time.sleep(0.5)  # Rate limiting
        try:
            history = str(market.get_kline(coin_choice, timeframe, startAt=end_time, endAt=start_time)).replace(']]', '], ').replace('[[', '[').split('], [')
        except Exception as e:
            PrintException()
            time.sleep(3.5)
            continue

        # Add fetched candles to history
        for item in history:
            history_list.append(item)

        current_change = len(history)
        vprint(f"Fetched {current_change} candles, total: {len(history_list)}")

        # Stop if we've fetched fewer than 1000 candles (reached end of available data)
        if current_change < 1000:
            break

        # Move time window backwards
        start_time = end_time
        end_time = int(start_time - ((1500 * timeframe_minutes) * 60))

        # Stop if we've collected enough data
        if len(history_list) >= how_far_to_look_back:
            break

    print(f"Fetched {len(history_list)} candles for {tf_choice}")

    # Parse candles and filter by TRAIN_UNTIL_TS
    price_list = []
    high_price_list = []
    low_price_list = []
    open_price_list = []
    candles_filtered = 0

    for i, candle_str in enumerate(history_list):
        try:
            parts = str(candle_str).replace('"', '').replace("'", "").split(", ")
            candle_time = float(parts[0].replace('[', ''))

            # CRITICAL: Filter out any candles after TRAIN_UNTIL_TS
            if candle_time >= TRAIN_UNTIL_TS:
                candles_filtered += 1
                continue

            openPrice = float(parts[1])
            closePrice = float(parts[2])
            highPrice = float(parts[3])
            lowPrice = float(parts[4])

            open_price_list.append(openPrice)
            price_list.append(closePrice)
            high_price_list.append(highPrice)
            low_price_list.append(lowPrice)
        except:
            PrintException()
            continue

    print(f"Filtered {candles_filtered} candles after train_until timestamp")
    print(f"Training on {len(price_list)} candles")

    # Reverse lists (oldest to newest)
    open_price_list.reverse()
    price_list.reverse()
    high_price_list.reverse()
    low_price_list.reverse()

    if len(price_list) < 100:
        print(f"ERROR: Not enough data for {tf_choice}. Skipping.")
        the_big_index += 1
        continue

    # Training loop
    loop_i = 0
    price_list_length = int(len(price_list) * 0.5)
    restarted_yet = 0

    print(f"Starting training loop for {tf_choice}...")

    while True:
        loop_i += 1

        # Check stop signal occasionally
        if should_stop_training(loop_i):
            print(f'Finished training {tf_choice}')
            flush_memory(tf_choice, force=True)
            break

        # Get current window of prices
        try:
            price_list2 = price_list[:price_list_length]
            high_price_list2 = high_price_list[:price_list_length]
            low_price_list2 = low_price_list[:price_list_length]
            open_price_list2 = open_price_list[:price_list_length]
        except:
            break

        if len(price_list2) < 10:
            break

        # Calculate price changes
        price_change_list = []
        for idx in range(len(price_list2)):
            price_change = 100 * ((price_list2[idx] - open_price_list2[idx]) / open_price_list2[idx])
            price_change_list.append(price_change)

        high_price_change_list = []
        for idx in range(len(high_price_list2)):
            high_price_change = 100 * ((high_price_list2[idx] - open_price_list2[idx]) / open_price_list2[idx])
            high_price_change_list.append(high_price_change)

        low_price_change_list = []
        for idx in range(len(low_price_list2)):
            low_price_change = 100 * ((low_price_list2[idx] - open_price_list2[idx]) / open_price_list2[idx])
            low_price_change_list.append(low_price_change)

        # Pattern matching logic (simplified from original trainer)
        current_pattern_length = number_of_candles[0]
        if len(price_change_list) < current_pattern_length + 1:
            break

        # Build current pattern
        current_pattern = price_change_list[-(current_pattern_length - 1):]

        # Load memories and find matches
        perfect_threshold = 1.0
        try:
            threshold_path = os.path.join(OUTPUT_DIR, f"neural_perfect_threshold_{tf_choice}.txt")
            with open(threshold_path, 'r') as f:
                perfect_threshold = float(f.read().strip())
        except:
            perfect_threshold = 1.0

        # Pattern matching with memories
        moves = []
        move_weights = []
        high_moves = []
        low_moves = []
        high_move_weights = []
        low_move_weights = []
        perfect_dexs = []

        for mem_ind, mem_entry in enumerate(memory_list):
            if not mem_entry or mem_entry.strip() == "":
                continue

            try:
                memory_pattern = mem_entry.split('{}')[0].replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')

                if len(memory_pattern) < len(current_pattern):
                    continue

                # Calculate pattern similarity
                checks = []
                for check_dex in range(len(current_pattern)):
                    current_candle = float(current_pattern[check_dex])
                    memory_candle = float(memory_pattern[check_dex])

                    if current_candle + memory_candle == 0.0:
                        difference = 0.0
                    else:
                        try:
                            difference = abs((abs(current_candle - memory_candle) / ((current_candle + memory_candle) / 2)) * 100)
                        except:
                            difference = 0.0
                    checks.append(difference)

                diff_avg = sum(checks) / len(checks) if checks else 100.0

                # If pattern matches, use it for prediction
                if diff_avg <= perfect_threshold:
                    high_diff = float(mem_entry.split('{}')[1]) / 100
                    low_diff = float(mem_entry.split('{}')[2]) / 100

                    moves.append(float(memory_pattern[-1]) * float(weight_list[mem_ind]))
                    move_weights.append(float(weight_list[mem_ind]))
                    high_moves.append(high_diff * float(high_weight_list[mem_ind]))
                    high_move_weights.append(float(high_weight_list[mem_ind]))
                    low_moves.append(low_diff * float(low_weight_list[mem_ind]))
                    low_move_weights.append(float(low_weight_list[mem_ind]))
                    perfect_dexs.append(mem_ind)
            except:
                continue

        # Adjust threshold based on matches
        if len(moves) > 20:
            perfect_threshold = max(0.0, perfect_threshold - 0.01)
        else:
            perfect_threshold = min(100.0, perfect_threshold + 0.01)

        write_threshold_sometimes(tf_choice, perfect_threshold, loop_i, every=200)

        # Calculate prediction
        if moves:
            final_move = sum(moves) / len(moves)
            high_final_move = sum(high_moves) / len(high_moves)
            low_final_move = sum(low_moves) / len(low_moves)
        else:
            final_move = 0.0
            high_final_move = 0.0
            low_final_move = 0.0

        # Update weights based on actual outcome
        if price_list_length >= len(price_list):
            # Reached end of data
            break

        price_list_length += 1

        # Get actual next price
        actual_price = price_list2[-1] if price_list2 else 0.0
        next_price = price_list[price_list_length - 1] if price_list_length < len(price_list) else actual_price
        high_next_price = high_price_list[price_list_length - 1] if price_list_length < len(high_price_list) else actual_price
        low_next_price = low_price_list[price_list_length - 1] if price_list_length < len(low_price_list) else actual_price

        # Calculate actual move
        actual_move_pct = ((next_price - actual_price) / abs(actual_price)) * 100 if actual_price != 0 else 0.0
        high_actual_move_pct = ((high_next_price - actual_price) / abs(actual_price)) * 100 if actual_price != 0 else 0.0
        low_actual_move_pct = ((low_next_price - actual_price) / abs(actual_price)) * 100 if actual_price != 0 else 0.0

        # Update weights for matched patterns
        for i, mem_ind in enumerate(perfect_dexs):
            if mem_ind >= len(weight_list):
                continue

            predicted_move = moves[i] / move_weights[i] if move_weights[i] != 0 else 0.0
            high_predicted_move = high_moves[i] / high_move_weights[i] if high_move_weights[i] != 0 else 0.0
            low_predicted_move = low_moves[i] / low_move_weights[i] if low_move_weights[i] != 0 else 0.0

            # Adjust weights based on prediction accuracy
            if actual_move_pct > predicted_move * 1.1:
                new_weight = min(2.0, float(weight_list[mem_ind]) + 0.25)
            elif actual_move_pct < predicted_move * 0.9:
                new_weight = max(0.0, float(weight_list[mem_ind]) - 0.25)
            else:
                new_weight = float(weight_list[mem_ind])

            weight_list[mem_ind] = str(new_weight)

            # Update high weights
            if high_actual_move_pct > high_predicted_move * 1.1:
                new_high_weight = min(2.0, float(high_weight_list[mem_ind]) + 0.25)
            elif high_actual_move_pct < high_predicted_move * 0.9:
                new_high_weight = max(0.0, float(high_weight_list[mem_ind]) - 0.25)
            else:
                new_high_weight = float(high_weight_list[mem_ind])

            high_weight_list[mem_ind] = str(new_high_weight)

            # Update low weights
            if low_actual_move_pct < low_predicted_move * 0.9:
                new_low_weight = min(2.0, float(low_weight_list[mem_ind]) + 0.25)
            elif low_actual_move_pct > low_predicted_move * 1.1:
                new_low_weight = max(0.0, float(low_weight_list[mem_ind]) - 0.25)
            else:
                new_low_weight = float(low_weight_list[mem_ind])

            low_weight_list[mem_ind] = str(new_low_weight)

            _mem = load_memory(tf_choice)
            _mem["weight_list"] = weight_list
            _mem["high_weight_list"] = high_weight_list
            _mem["low_weight_list"] = low_weight_list
            _mem["dirty"] = True

        # Add new memory if no good matches
        if not moves:
            new_pattern = current_pattern.copy()
            new_pattern.append(actual_move_pct)
            mem_entry = ' '.join([str(x) for x in new_pattern]) + '{}' + str(high_actual_move_pct) + '{}' + str(low_actual_move_pct)

            _mem = load_memory(tf_choice)
            _mem["memory_list"].append(mem_entry)
            _mem["weight_list"].append('1.0')
            _mem["high_weight_list"].append('1.0')
            _mem["low_weight_list"].append('1.0')
            _mem["dirty"] = True

            memory_list = _mem["memory_list"]
            weight_list = _mem["weight_list"]
            high_weight_list = _mem["high_weight_list"]
            low_weight_list = _mem["low_weight_list"]

        # Periodic flush
        if loop_i % 200 == 0:
            flush_memory(tf_choice)
            vprint(f"Loop {loop_i}: {len(memory_list)} memories, threshold={perfect_threshold:.3f}, candle {price_list_length}/{len(price_list)}")

    # Flush final state
    flush_memory(tf_choice, force=True)
    print(f"Completed training for {tf_choice}")

    the_big_index += 1

# Mark training finished
_trainer_finished_at = int(time.time())
try:
    last_training_path = os.path.join(OUTPUT_DIR, "trainer_last_training_time.txt")
    with open(last_training_path, 'w+') as f:
        f.write(str(_trainer_finished_at))
except:
    pass

try:
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coin": _arg_coin,
                "state": "FINISHED",
                "started_at": _trainer_started_at,
                "finished_at": _trainer_finished_at,
                "timestamp": _trainer_finished_at,
                "train_until_ts": TRAIN_UNTIL_TS,
                "mode": "incremental"
            },
            f,
        )
except Exception:
    pass

print("\n" + "=" * 60)
print("INCREMENTAL TRAINING COMPLETED")
print("=" * 60)
print(f"Coin: {_arg_coin}")
print(f"Training limited to: {datetime.datetime.fromtimestamp(TRAIN_UNTIL_TS)}")
print(f"Started: {datetime.datetime.fromtimestamp(_trainer_started_at)}")
print(f"Finished: {datetime.datetime.fromtimestamp(_trainer_finished_at)}")
print(f"Duration: {(_trainer_finished_at - _trainer_started_at) / 60:.1f} minutes")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)

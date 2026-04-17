//+------------------------------------------------------------------+
//|             AzirFractalCandidateEventExport.mq5                  |
//|  Auxiliary exporter for baseline_azir_setup_candidate_fractal_v1 |
//+------------------------------------------------------------------+
#property strict
#property script_show_inputs

input string   InpSymbol                     = "XAUUSD-STD";
input datetime InpStartTime                  = D'2021.01.01 00:00:00';
input datetime InpEndTime                    = D'2025.12.31 23:59:59';
input string   InpFileName                   = "fractal_candidate_event_log.csv";
input bool     InpUseCommonFolder            = true;
input bool     InpOverwrite                  = true;
input int      InpSetupHour                  = 16;
input int      InpSetupMinute                = 30;
input int      InpCloseHour                  = 22;
input int      InpSwingBars                  = 10;
input int      InpFractalSideBars            = 2;
input double   InpEntryOffsetPoints          = 5.0;
input long     InpMagic                      = 123456321;
input double   InpLotSize                    = 0.10;
input int      InpStopLossPoints             = 500;
input int      InpTakeProfitPoints           = 500;
input int      InpTrailingStartPoints        = 90;
input int      InpTrailingStepPoints         = 50;
input bool     InpAllowBuys                  = true;
input bool     InpAllowSells                 = true;
input bool     InpTrendFilterEnabled         = true;
input bool     InpATRFilterEnabled           = true;
input int      InpATRPeriod                  = 14;
input double   InpATRMinimumPoints           = 100.0;
input bool     InpRSIGateEnabled             = true;
input int      InpRSIPeriod                  = 14;
input double   InpRSIBullishThreshold        = 70.0;
input double   InpRSISellThreshold           = 30.0;
input double   InpMinDistanceBetweenPendings = 200.0;
input bool     InpNoTradeFridays             = true;
input bool     InpPrintProgress              = true;

string BoolText(const bool value)
{
   return value ? "true" : "false";
}

string LongText(const long value)
{
   return StringFormat("%I64d", value);
}

string IntText(const int value)
{
   return IntegerToString(value);
}

string DoubleText(const double value, const int digits)
{
   return DoubleToString(value, digits);
}

string FormatTimestamp(const datetime value)
{
   return TimeToString(value, TIME_DATE | TIME_SECONDS);
}

string FormatDateForId(const datetime value)
{
   string text = TimeToString(value, TIME_DATE);
   StringReplace(text, ".", "-");
   return text;
}

int DayOfWeekMql(const datetime value)
{
   MqlDateTime parts;
   TimeToStruct(value, parts);
   return parts.day_of_week;
}

bool IsSetupTime(const datetime value)
{
   MqlDateTime parts;
   TimeToStruct(value, parts);
   return parts.hour == InpSetupHour && parts.min == InpSetupMinute;
}

bool OpenOutputFile(const string file_name, int &handle)
{
   int flags = FILE_WRITE | FILE_TXT | FILE_ANSI;
   if(InpUseCommonFolder)
      flags |= FILE_COMMON;

   if(!InpOverwrite)
   {
      const int common_flag = InpUseCommonFolder ? FILE_COMMON : 0;
      if(FileIsExist(file_name, common_flag))
      {
         Print("Error: output file already exists and InpOverwrite=false. File: ", file_name);
         return false;
      }
   }

   handle = FileOpen(file_name, flags);
   if(handle == INVALID_HANDLE)
   {
      Print("Error opening output file: ", file_name, " | LastError=", GetLastError());
      return false;
   }

   return true;
}

string CsvCell(const string value)
{
   string escaped = value;
   StringReplace(escaped, "\"", "\"\"");
   return "\"" + escaped + "\"";
}

void AddCell(string &cells[], const string value)
{
   const int size = ArraySize(cells);
   ArrayResize(cells, size + 1);
   cells[size] = value;
}

void WriteCsvCells(const int handle, string &cells[])
{
   string line = "";
   const int size = ArraySize(cells);
   for(int index = 0; index < size; index++)
   {
      if(index > 0)
         line += ",";
      line += CsvCell(cells[index]);
   }
   FileWriteString(handle, line + "\r\n");
}

void WriteHeader(const int handle)
{
   string cells[];
   AddCell(cells, "timestamp");
   AddCell(cells, "event_id");
   AddCell(cells, "event_type");
   AddCell(cells, "symbol");
   AddCell(cells, "magic");
   AddCell(cells, "day_of_week");
   AddCell(cells, "is_friday");
   AddCell(cells, "server_time");
   AddCell(cells, "broker");
   AddCell(cells, "account");
   AddCell(cells, "timeframe");
   AddCell(cells, "ny_open_hour");
   AddCell(cells, "ny_open_minute");
   AddCell(cells, "close_hour");
   AddCell(cells, "swing_bars");
   AddCell(cells, "lot_size");
   AddCell(cells, "sl_points");
   AddCell(cells, "tp_points");
   AddCell(cells, "trailing_start_points");
   AddCell(cells, "trailing_step_points");
   AddCell(cells, "swing_high");
   AddCell(cells, "swing_low");
   AddCell(cells, "buy_entry");
   AddCell(cells, "sell_entry");
   AddCell(cells, "pending_distance_points");
   AddCell(cells, "spread_points");
   AddCell(cells, "ema20");
   AddCell(cells, "prev_close");
   AddCell(cells, "prev_close_vs_ema20_points");
   AddCell(cells, "prev_close_above_ema20");
   AddCell(cells, "atr");
   AddCell(cells, "atr_points");
   AddCell(cells, "atr_filter_enabled");
   AddCell(cells, "atr_filter_passed");
   AddCell(cells, "atr_minimum");
   AddCell(cells, "rsi");
   AddCell(cells, "rsi_gate_enabled");
   AddCell(cells, "rsi_gate_required");
   AddCell(cells, "rsi_gate_passed");
   AddCell(cells, "rsi_bullish_threshold");
   AddCell(cells, "rsi_sell_threshold");
   AddCell(cells, "allow_buys");
   AddCell(cells, "allow_sells");
   AddCell(cells, "trend_filter_enabled");
   AddCell(cells, "buy_allowed_by_trend");
   AddCell(cells, "sell_allowed_by_trend");
   AddCell(cells, "buy_order_placed");
   AddCell(cells, "sell_order_placed");
   AddCell(cells, "buy_retcode");
   AddCell(cells, "sell_retcode");
   AddCell(cells, "fill_side");
   AddCell(cells, "fill_price");
   AddCell(cells, "duration_to_fill_seconds");
   AddCell(cells, "mfe_points");
   AddCell(cells, "mae_points");
   AddCell(cells, "exit_reason");
   AddCell(cells, "gross_pnl");
   AddCell(cells, "net_pnl");
   AddCell(cells, "commission");
   AddCell(cells, "swap");
   AddCell(cells, "slippage_points");
   AddCell(cells, "trailing_activated");
   AddCell(cells, "trailing_modifications");
   AddCell(cells, "trailing_outcome");
   AddCell(cells, "opposite_order_cancelled");
   AddCell(cells, "notes");
   WriteCsvCells(handle, cells);
}

void WriteCanonicalRow(
   const int handle,
   const string sym,
   const int digits,
   const datetime setup_time,
   const string event_type,
   const int day_of_week,
   const bool is_friday,
   const string swing_high,
   const string swing_low,
   const string buy_entry,
   const string sell_entry,
   const string pending_distance_points,
   const string ema20,
   const string prev_close,
   const string prev_close_vs_ema20_points,
   const string prev_close_above_ema20,
   const string atr,
   const string atr_points,
   const string atr_filter_passed,
   const string rsi,
   const string rsi_gate_required,
   const string rsi_gate_passed,
   const string buy_allowed_by_trend,
   const string sell_allowed_by_trend,
   const string buy_order_placed,
   const string sell_order_placed,
   const string buy_retcode,
   const string sell_retcode,
   const string notes
)
{
   const string timestamp = FormatTimestamp(setup_time);
   const string event_id = FormatDateForId(setup_time) + "_" + sym + "_" + LongText(InpMagic);

   string cells[];
   AddCell(cells, timestamp);
   AddCell(cells, event_id);
   AddCell(cells, event_type);
   AddCell(cells, sym);
   AddCell(cells, LongText(InpMagic));
   AddCell(cells, IntText(day_of_week));
   AddCell(cells, BoolText(is_friday));
   AddCell(cells, timestamp);
   AddCell(cells, AccountInfoString(ACCOUNT_COMPANY));
   AddCell(cells, LongText(AccountInfoInteger(ACCOUNT_LOGIN)));
   AddCell(cells, "M5");
   AddCell(cells, IntText(InpSetupHour));
   AddCell(cells, IntText(InpSetupMinute));
   AddCell(cells, IntText(InpCloseHour));
   AddCell(cells, IntText(InpSwingBars));
   AddCell(cells, DoubleText(InpLotSize, 2));
   AddCell(cells, IntText(InpStopLossPoints));
   AddCell(cells, IntText(InpTakeProfitPoints));
   AddCell(cells, IntText(InpTrailingStartPoints));
   AddCell(cells, IntText(InpTrailingStepPoints));
   AddCell(cells, swing_high);
   AddCell(cells, swing_low);
   AddCell(cells, buy_entry);
   AddCell(cells, sell_entry);
   AddCell(cells, pending_distance_points);
   AddCell(cells, "");
   AddCell(cells, ema20);
   AddCell(cells, prev_close);
   AddCell(cells, prev_close_vs_ema20_points);
   AddCell(cells, prev_close_above_ema20);
   AddCell(cells, atr);
   AddCell(cells, atr_points);
   AddCell(cells, BoolText(InpATRFilterEnabled));
   AddCell(cells, atr_filter_passed);
   AddCell(cells, DoubleText(InpATRMinimumPoints, 2));
   AddCell(cells, rsi);
   AddCell(cells, BoolText(InpRSIGateEnabled));
   AddCell(cells, rsi_gate_required);
   AddCell(cells, rsi_gate_passed);
   AddCell(cells, DoubleText(InpRSIBullishThreshold, 2));
   AddCell(cells, DoubleText(InpRSISellThreshold, 2));
   AddCell(cells, BoolText(InpAllowBuys));
   AddCell(cells, BoolText(InpAllowSells));
   AddCell(cells, BoolText(InpTrendFilterEnabled));
   AddCell(cells, buy_allowed_by_trend);
   AddCell(cells, sell_allowed_by_trend);
   AddCell(cells, buy_order_placed);
   AddCell(cells, sell_order_placed);
   AddCell(cells, buy_retcode);
   AddCell(cells, sell_retcode);
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "");
   AddCell(cells, "false");
   AddCell(cells, "0");
   AddCell(cells, "");
   AddCell(cells, "false");
   AddCell(cells, notes);
   WriteCsvCells(handle, cells);
}

bool RollingHigh(MqlRates &rates[], const int setup_index, const int lookback, double &value)
{
   if(setup_index < lookback)
      return false;

   value = rates[setup_index - lookback].high;
   for(int index = setup_index - lookback + 1; index < setup_index; index++)
   {
      if(rates[index].high > value)
         value = rates[index].high;
   }
   return true;
}

bool RollingLow(MqlRates &rates[], const int setup_index, const int lookback, double &value)
{
   if(setup_index < lookback)
      return false;

   value = rates[setup_index - lookback].low;
   for(int index = setup_index - lookback + 1; index < setup_index; index++)
   {
      if(rates[index].low < value)
         value = rates[index].low;
   }
   return true;
}

bool FindLastPivotHigh(MqlRates &rates[], const int setup_index, const int lookback, const int side, double &value)
{
   if(setup_index < lookback || side < 1)
      return false;

   int start = setup_index - lookback;
   if(start < 0)
      start = 0;

   const int first_candidate = start + side;
   const int last_candidate = setup_index - side - 1;
   if(last_candidate < first_candidate)
      return false;

   for(int index = last_candidate; index >= first_candidate; index--)
   {
      const double candidate = rates[index].high;
      bool pivot = true;

      for(int left = index - side; left < index; left++)
      {
         if(candidate <= rates[left].high)
         {
            pivot = false;
            break;
         }
      }

      if(!pivot)
         continue;

      for(int right = index + 1; right <= index + side; right++)
      {
         if(candidate <= rates[right].high)
         {
            pivot = false;
            break;
         }
      }

      if(pivot)
      {
         value = candidate;
         return true;
      }
   }

   return false;
}

bool FindLastPivotLow(MqlRates &rates[], const int setup_index, const int lookback, const int side, double &value)
{
   if(setup_index < lookback || side < 1)
      return false;

   int start = setup_index - lookback;
   if(start < 0)
      start = 0;

   const int first_candidate = start + side;
   const int last_candidate = setup_index - side - 1;
   if(last_candidate < first_candidate)
      return false;

   for(int index = last_candidate; index >= first_candidate; index--)
   {
      const double candidate = rates[index].low;
      bool pivot = true;

      for(int left = index - side; left < index; left++)
      {
         if(candidate >= rates[left].low)
         {
            pivot = false;
            break;
         }
      }

      if(!pivot)
         continue;

      for(int right = index + 1; right <= index + side; right++)
      {
         if(candidate >= rates[right].low)
         {
            pivot = false;
            break;
         }
      }

      if(pivot)
      {
         value = candidate;
         return true;
      }
   }

   return false;
}

bool ComputeFractalSwingHigh(MqlRates &rates[], const int setup_index, double &value)
{
   if(FindLastPivotHigh(rates, setup_index, InpSwingBars, InpFractalSideBars, value))
      return true;
   return RollingHigh(rates, setup_index, InpSwingBars, value);
}

bool ComputeFractalSwingLow(MqlRates &rates[], const int setup_index, double &value)
{
   if(FindLastPivotLow(rates, setup_index, InpSwingBars, InpFractalSideBars, value))
      return true;
   return RollingLow(rates, setup_index, InpSwingBars, value);
}

bool ComputeEMA(MqlRates &rates[], const int end_index, const int period, double &value)
{
   if(end_index < 0 || period <= 0)
      return false;

   const double alpha = 2.0 / ((double)period + 1.0);
   value = rates[0].close;
   for(int index = 1; index <= end_index; index++)
      value = rates[index].close * alpha + value * (1.0 - alpha);

   return true;
}

bool ComputeATR(MqlRates &rates[], const int end_index, const int period, double &value)
{
   if(end_index < period || period <= 0)
      return false;

   double total = 0.0;
   for(int index = end_index - period + 1; index <= end_index; index++)
   {
      const double high_low = rates[index].high - rates[index].low;
      const double high_close = MathAbs(rates[index].high - rates[index - 1].close);
      const double low_close = MathAbs(rates[index].low - rates[index - 1].close);
      total += MathMax(high_low, MathMax(high_close, low_close));
   }

   value = total / (double)period;
   return true;
}

double RSIFromAverages(const double avg_gain, const double avg_loss)
{
   if(avg_loss == 0.0)
      return 100.0;

   const double rs = avg_gain / avg_loss;
   return 100.0 - (100.0 / (1.0 + rs));
}

bool ComputeRSI(MqlRates &rates[], const int end_index, const int period, double &value)
{
   if(end_index < period || period <= 0)
      return false;

   double avg_gain = 0.0;
   double avg_loss = 0.0;
   for(int index = 1; index <= period; index++)
   {
      const double change = rates[index].close - rates[index - 1].close;
      if(change >= 0.0)
         avg_gain += change;
      else
         avg_loss += -change;
   }

   avg_gain /= (double)period;
   avg_loss /= (double)period;
   value = RSIFromAverages(avg_gain, avg_loss);

   for(int index = period + 1; index <= end_index; index++)
   {
      const double change = rates[index].close - rates[index - 1].close;
      const double gain = change > 0.0 ? change : 0.0;
      const double loss = change < 0.0 ? -change : 0.0;
      avg_gain = ((avg_gain * (double)(period - 1)) + gain) / (double)period;
      avg_loss = ((avg_loss * (double)(period - 1)) + loss) / (double)period;
      value = RSIFromAverages(avg_gain, avg_loss);
   }

   return true;
}

void OnStart()
{
   const string sym = InpSymbol == "" ? _Symbol : InpSymbol;

   if(InpEndTime <= InpStartTime)
   {
      Print("Error: InpEndTime must be greater than InpStartTime.");
      return;
   }

   if(InpSwingBars < (InpFractalSideBars * 2 + 1))
   {
      Print("Error: InpSwingBars is too small for the requested fractal side bars.");
      return;
   }

   if(!SymbolSelect(sym, true))
   {
      Print("Error: could not select symbol ", sym);
      return;
   }

   const int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   const double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   if(point <= 0.0)
   {
      Print("Error: symbol point is not available for ", sym);
      return;
   }

   int handle = INVALID_HANDLE;
   if(!OpenOutputFile(InpFileName, handle))
      return;

   WriteHeader(handle);

   const datetime history_start = InpStartTime - 60 * 86400;
   MqlRates rates[];
   ResetLastError();
   const int copied = CopyRates(sym, PERIOD_M5, history_start, InpEndTime, rates);
   if(copied <= 0)
   {
      Print("Error: CopyRates returned no M5 bars. LastError=", GetLastError());
      FileClose(handle);
      return;
   }
   ArraySetAsSeries(rates, false);

   int written = 0;
   int skipped_history = 0;
   const int required_history = MathMax(20, MathMax(InpSwingBars, MathMax(InpATRPeriod, InpRSIPeriod))) + 1;

   for(int index = 0; index < copied; index++)
   {
      const datetime setup_time = rates[index].time;
      if(setup_time < InpStartTime || setup_time > InpEndTime)
         continue;
      if(!IsSetupTime(setup_time))
         continue;

      const int day_of_week = DayOfWeekMql(setup_time);
      const bool is_friday = day_of_week == 5;

      if(InpNoTradeFridays && is_friday)
      {
         WriteCanonicalRow(
            handle,
            sym,
            digits,
            setup_time,
            "blocked_friday",
            day_of_week,
            true,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            BoolText(InpATRFilterEnabled),
            "",
            "false",
            "true",
            "false",
            "false",
            "false",
            "false",
            "0",
            "0",
            "NoTradeFridays blocked the daily opportunity before order evaluation."
         );
         written++;
         continue;
      }

      if(index < required_history)
      {
         skipped_history++;
         continue;
      }

      const int previous_index = index - 1;
      double swing_high = 0.0;
      double swing_low = 0.0;
      double ema20 = 0.0;
      double atr = 0.0;
      double rsi = 0.0;

      if(!ComputeFractalSwingHigh(rates, index, swing_high))
         continue;
      if(!ComputeFractalSwingLow(rates, index, swing_low))
         continue;
      if(!ComputeEMA(rates, previous_index, 20, ema20))
         continue;
      if(!ComputeATR(rates, previous_index, InpATRPeriod, atr))
         continue;

      const bool has_rsi = ComputeRSI(rates, previous_index, InpRSIPeriod, rsi);
      const double prev_close = rates[previous_index].close;
      const double atr_points = atr / point;
      const double buy_entry = swing_high + InpEntryOffsetPoints * point;
      const double sell_entry = swing_low - InpEntryOffsetPoints * point;
      const double pending_distance_points = (buy_entry - sell_entry) / point;
      const double prev_close_vs_ema20_points = (prev_close - ema20) / point;
      const bool prev_close_above_ema20 = prev_close > ema20;
      const bool buy_allowed_by_trend = (!InpTrendFilterEnabled || prev_close > ema20) && InpAllowBuys;
      const bool sell_allowed_by_trend = (!InpTrendFilterEnabled || prev_close < ema20) && InpAllowSells;
      const bool atr_filter_passed = !InpATRFilterEnabled || atr_points >= InpATRMinimumPoints;

      bool buy_order_placed = false;
      bool sell_order_placed = false;
      string notes = "";

      if(atr_filter_passed)
      {
         if(InpTrendFilterEnabled)
         {
            if(prev_close > ema20 && InpAllowBuys)
               buy_order_placed = true;
            else if(prev_close < ema20 && InpAllowSells)
               sell_order_placed = true;
         }
         else
         {
            buy_order_placed = InpAllowBuys;
            sell_order_placed = InpAllowSells;
         }
      }

      const bool rsi_gate_required =
         InpRSIGateEnabled &&
         buy_order_placed &&
         sell_order_placed &&
         pending_distance_points >= InpMinDistanceBetweenPendings;

      if(!atr_filter_passed)
         notes = "ATR filter failed; no orders were sent.";
      else if(rsi_gate_required)
         notes = "Fractal candidate setup evaluated; RSI gate required because both sides were placed and distance threshold passed.";
      else
         notes = "Fractal candidate setup evaluated; RSI gate inactive for this opportunity.";

      WriteCanonicalRow(
         handle,
         sym,
         digits,
         setup_time,
         "opportunity",
         day_of_week,
         is_friday,
         DoubleText(swing_high, digits),
         DoubleText(swing_low, digits),
         DoubleText(buy_entry, digits),
         DoubleText(sell_entry, digits),
         DoubleText(pending_distance_points, 2),
         DoubleText(ema20, digits),
         DoubleText(prev_close, digits),
         DoubleText(prev_close_vs_ema20_points, 2),
         BoolText(prev_close_above_ema20),
         DoubleText(atr, digits),
         DoubleText(atr_points, 2),
         BoolText(atr_filter_passed),
         has_rsi ? DoubleText(rsi, 2) : "",
         BoolText(rsi_gate_required),
         "true",
         BoolText(buy_allowed_by_trend),
         BoolText(sell_allowed_by_trend),
         BoolText(buy_order_placed),
         BoolText(sell_order_placed),
         buy_order_placed ? "10008" : "0",
         sell_order_placed ? "10008" : "0",
         notes
      );
      written++;

      if(InpPrintProgress && written % 250 == 0)
         Print("Fractal candidate rows exported: ", written, " | last setup=", FormatTimestamp(setup_time));
   }

   FileClose(handle);

   const string folder_info = InpUseCommonFolder
                            ? TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\"
                            : TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";

   Print("======================================");
   Print("Azir fractal candidate export completed.");
   Print("Symbol: ", sym);
   Print("Rows written: ", written);
   Print("Skipped early setup rows due to insufficient history: ", skipped_history);
   Print("File: ", InpFileName);
   Print("Folder: ", folder_info);
   Print("Candidate: baseline_azir_setup_candidate_fractal_v1");
   Print("Definition: swing_10_fractal, 2-left/2-right pivot, rolling fallback.");
   Print("======================================");
}

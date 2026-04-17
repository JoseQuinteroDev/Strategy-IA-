//+------------------------------------------------------------------+
//| AzirIA MT5 event logger                                          |
//| Append-only CSV logging for research replication and audit.       |
//+------------------------------------------------------------------+
#property strict

#ifndef AZIR_EVENT_LOGGER_MQH
#define AZIR_EVENT_LOGGER_MQH

bool   g_azir_log_enabled      = true;
bool   g_azir_log_common       = true;
string g_azir_log_file_name    = "";
int    g_azir_log_handle       = INVALID_HANDLE;
string g_azir_log_symbol       = "";
int    g_azir_log_magic        = 0;
string g_azir_log_event_id     = "";

datetime g_azir_setup_time             = 0;
datetime g_azir_fill_time              = 0;
long     g_azir_position_id            = 0;
int      g_azir_fill_side              = 0; // 1=buy, -1=sell
double   g_azir_buy_entry              = 0.0;
double   g_azir_sell_entry             = 0.0;
double   g_azir_fill_price             = 0.0;
double   g_azir_mfe_points             = 0.0;
double   g_azir_mae_points             = 0.0;
bool     g_azir_trailing_activated     = false;
int      g_azir_trailing_modifications = 0;
bool     g_azir_opposite_cancelled     = false;
datetime g_azir_last_friday_log_day    = 0;
datetime g_azir_last_no_fill_log_day   = 0;

void AzirLogConfigure(const bool enabled, const string file_name, const bool use_common_folder)
{
   g_azir_log_enabled   = enabled;
   g_azir_log_file_name = file_name;
   g_azir_log_common    = use_common_folder;
}

string AzirLogBool(const bool value)
{
   return value ? "true" : "false";
}

string AzirLogDouble(const double value, const int digits = 5)
{
   if(!MathIsValidNumber(value) || value == EMPTY_VALUE)
      return "";
   return DoubleToString(value, digits);
}

string AzirLogTime(const datetime value)
{
   if(value <= 0)
      return "";
   return TimeToString(value, TIME_DATE | TIME_SECONDS);
}

string AzirLogDealSide(const long deal_type)
{
   if(deal_type == DEAL_TYPE_BUY)
      return "buy";
   if(deal_type == DEAL_TYPE_SELL)
      return "sell";
   return "unknown";
}

string AzirLogExitReason(const long deal_reason)
{
   switch((ENUM_DEAL_REASON)deal_reason)
   {
      case DEAL_REASON_SL:       return "stop_loss_or_trailing_stop";
      case DEAL_REASON_TP:       return "take_profit";
      case DEAL_REASON_SO:       return "stop_out";
      case DEAL_REASON_CLIENT:   return "manual_close";
      case DEAL_REASON_EXPERT:   return "expert_close_or_session_close";
      case DEAL_REASON_MOBILE:   return "mobile_close";
      case DEAL_REASON_WEB:      return "web_close";
      default:                   return "unknown";
   }
}

string AzirLogBuildDefaultFileName(const string symbol, const int magic)
{
   return "azir_events_" + symbol + "_" + IntegerToString(magic) + ".csv";
}

string AzirLogEscapeCsv(string value)
{
   bool must_quote = (StringFind(value, ",") >= 0 ||
                      StringFind(value, "\"") >= 0 ||
                      StringFind(value, "\r") >= 0 ||
                      StringFind(value, "\n") >= 0);
   if(StringFind(value, "\"") >= 0)
      StringReplace(value, "\"", "\"\"");
   if(must_quote)
      return "\"" + value + "\"";
   return value;
}

void AzirLogAppendCsvField(string &line, const string value)
{
   if(StringLen(line) > 0)
      line += ",";
   line += AzirLogEscapeCsv(value);
}

void AzirLogWriteCsvLine(string &fields[])
{
   string line = "";
   int count = ArraySize(fields);
   for(int i=0; i<count; ++i)
      AzirLogAppendCsvField(line, fields[i]);
   FileWriteString(g_azir_log_handle, line + "\r\n");
}

void AzirLogAddCsvField(string &fields[], const string value)
{
   int index = ArraySize(fields);
   ArrayResize(fields, index + 1);
   fields[index] = value;
}

bool AzirLogOpenIfNeeded()
{
   if(!g_azir_log_enabled)
      return false;
   if(g_azir_log_handle != INVALID_HANDLE)
      return true;

   string file_name = g_azir_log_file_name;
   if(file_name == "")
      file_name = AzirLogBuildDefaultFileName(g_azir_log_symbol, g_azir_log_magic);

   int common_flag = g_azir_log_common ? FILE_COMMON : 0;
   bool existed = FileIsExist(file_name, common_flag);
   int flags = FILE_READ | FILE_WRITE | FILE_ANSI;
   if(g_azir_log_common)
      flags |= FILE_COMMON;

   g_azir_log_handle = FileOpen(file_name, flags, ',');
   if(g_azir_log_handle == INVALID_HANDLE)
   {
      Print("Azir event logger could not open file: ", file_name, " LastError=", GetLastError());
      return false;
   }

   if(!existed || FileSize(g_azir_log_handle) == 0)
   {
      FileWriteString(
         g_azir_log_handle,
         "timestamp,event_id,event_type,symbol,magic,day_of_week,is_friday,server_time,broker,account,"
         "timeframe,ny_open_hour,ny_open_minute,close_hour,swing_bars,lot_size,sl_points,tp_points,"
         "trailing_start_points,trailing_step_points,swing_high,swing_low,buy_entry,sell_entry,"
         "pending_distance_points,spread_points,ema20,prev_close,prev_close_vs_ema20_points,"
         "prev_close_above_ema20,atr,atr_points,atr_filter_enabled,atr_filter_passed,atr_minimum,"
         "rsi,rsi_gate_enabled,rsi_gate_required,rsi_gate_passed,rsi_bullish_threshold,rsi_sell_threshold,"
         "allow_buys,allow_sells,trend_filter_enabled,buy_allowed_by_trend,sell_allowed_by_trend,"
         "buy_order_placed,sell_order_placed,buy_retcode,sell_retcode,fill_side,fill_price,"
         "duration_to_fill_seconds,mfe_points,mae_points,exit_reason,gross_pnl,net_pnl,commission,swap,"
         "slippage_points,trailing_activated,trailing_modifications,trailing_outcome,opposite_order_cancelled,notes\r\n"
      );
   }
   FileSeek(g_azir_log_handle, 0, SEEK_END);
   return true;
}

void AzirLogInit(const string symbol, const int magic)
{
   g_azir_log_symbol = symbol;
   g_azir_log_magic  = magic;
   AzirLogOpenIfNeeded();
}

void AzirLogClose()
{
   if(g_azir_log_handle != INVALID_HANDLE)
   {
      FileFlush(g_azir_log_handle);
      FileClose(g_azir_log_handle);
      g_azir_log_handle = INVALID_HANDLE;
   }
}

void AzirLogResetDailyState()
{
   g_azir_setup_time             = 0;
   g_azir_fill_time              = 0;
   g_azir_position_id            = 0;
   g_azir_fill_side              = 0;
   g_azir_buy_entry              = 0.0;
   g_azir_sell_entry             = 0.0;
   g_azir_fill_price             = 0.0;
   g_azir_mfe_points             = 0.0;
   g_azir_mae_points             = 0.0;
   g_azir_trailing_activated     = false;
   g_azir_trailing_modifications = 0;
   g_azir_opposite_cancelled     = false;
   g_azir_log_event_id           = "";
}

void AzirLogWriteRow(
   const datetime timestamp,
   const string event_type,
   const int day_of_week,
   const bool is_friday,
   const string timeframe,
   const int ny_open_hour,
   const int ny_open_minute,
   const int close_hour,
   const int swing_bars,
   const double lot_size,
   const int sl_points,
   const int tp_points,
   const int trailing_start_points,
   const int trailing_step_points,
   const double swing_high,
   const double swing_low,
   const double buy_entry,
   const double sell_entry,
   const double pending_distance_points,
   const double spread_points,
   const double ema20,
   const double prev_close,
   const double prev_close_vs_ema20_points,
   const bool prev_close_above_ema20,
   const double atr,
   const double atr_points,
   const bool atr_filter_enabled,
   const bool atr_filter_passed,
   const double atr_minimum,
   const double rsi,
   const bool rsi_gate_enabled,
   const bool rsi_gate_required,
   const bool rsi_gate_passed,
   const double rsi_bullish_threshold,
   const double rsi_sell_threshold,
   const bool allow_buys,
   const bool allow_sells,
   const bool trend_filter_enabled,
   const bool buy_allowed_by_trend,
   const bool sell_allowed_by_trend,
   const bool buy_order_placed,
   const bool sell_order_placed,
   const uint buy_retcode,
   const uint sell_retcode,
   const string fill_side,
   const double fill_price,
   const long duration_to_fill_seconds,
   const double mfe_points,
   const double mae_points,
   const string exit_reason,
   const double gross_pnl,
   const double net_pnl,
   const double commission,
   const double swap,
   const double slippage_points,
   const bool trailing_activated,
   const int trailing_modifications,
   const string trailing_outcome,
   const bool opposite_order_cancelled,
   const string notes
)
{
   if(!AzirLogOpenIfNeeded())
      return;

   string fields[];
   AzirLogAddCsvField(fields, AzirLogTime(timestamp));
   AzirLogAddCsvField(fields, g_azir_log_event_id);
   AzirLogAddCsvField(fields, event_type);
   AzirLogAddCsvField(fields, g_azir_log_symbol);
   AzirLogAddCsvField(fields, IntegerToString(g_azir_log_magic));
   AzirLogAddCsvField(fields, IntegerToString(day_of_week));
   AzirLogAddCsvField(fields, AzirLogBool(is_friday));
   AzirLogAddCsvField(fields, AzirLogTime(timestamp));
   AzirLogAddCsvField(fields, AccountInfoString(ACCOUNT_COMPANY));
   AzirLogAddCsvField(fields, IntegerToString((int)AccountInfoInteger(ACCOUNT_LOGIN)));
   AzirLogAddCsvField(fields, timeframe);
   AzirLogAddCsvField(fields, IntegerToString(ny_open_hour));
   AzirLogAddCsvField(fields, IntegerToString(ny_open_minute));
   AzirLogAddCsvField(fields, IntegerToString(close_hour));
   AzirLogAddCsvField(fields, IntegerToString(swing_bars));
   AzirLogAddCsvField(fields, AzirLogDouble(lot_size, 2));
   AzirLogAddCsvField(fields, IntegerToString(sl_points));
   AzirLogAddCsvField(fields, IntegerToString(tp_points));
   AzirLogAddCsvField(fields, IntegerToString(trailing_start_points));
   AzirLogAddCsvField(fields, IntegerToString(trailing_step_points));
   AzirLogAddCsvField(fields, AzirLogDouble(swing_high, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(swing_low, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(buy_entry, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(sell_entry, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(pending_distance_points, 1));
   AzirLogAddCsvField(fields, AzirLogDouble(spread_points, 1));
   AzirLogAddCsvField(fields, AzirLogDouble(ema20, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(prev_close, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(prev_close_vs_ema20_points, 1));
   AzirLogAddCsvField(fields, AzirLogBool(prev_close_above_ema20));
   AzirLogAddCsvField(fields, AzirLogDouble(atr, 5));
   AzirLogAddCsvField(fields, AzirLogDouble(atr_points, 1));
   AzirLogAddCsvField(fields, AzirLogBool(atr_filter_enabled));
   AzirLogAddCsvField(fields, AzirLogBool(atr_filter_passed));
   AzirLogAddCsvField(fields, AzirLogDouble(atr_minimum, 1));
   AzirLogAddCsvField(fields, AzirLogDouble(rsi, 2));
   AzirLogAddCsvField(fields, AzirLogBool(rsi_gate_enabled));
   AzirLogAddCsvField(fields, AzirLogBool(rsi_gate_required));
   AzirLogAddCsvField(fields, AzirLogBool(rsi_gate_passed));
   AzirLogAddCsvField(fields, AzirLogDouble(rsi_bullish_threshold, 2));
   AzirLogAddCsvField(fields, AzirLogDouble(rsi_sell_threshold, 2));
   AzirLogAddCsvField(fields, AzirLogBool(allow_buys));
   AzirLogAddCsvField(fields, AzirLogBool(allow_sells));
   AzirLogAddCsvField(fields, AzirLogBool(trend_filter_enabled));
   AzirLogAddCsvField(fields, AzirLogBool(buy_allowed_by_trend));
   AzirLogAddCsvField(fields, AzirLogBool(sell_allowed_by_trend));
   AzirLogAddCsvField(fields, AzirLogBool(buy_order_placed));
   AzirLogAddCsvField(fields, AzirLogBool(sell_order_placed));
   AzirLogAddCsvField(fields, IntegerToString((int)buy_retcode));
   AzirLogAddCsvField(fields, IntegerToString((int)sell_retcode));
   AzirLogAddCsvField(fields, fill_side);
   AzirLogAddCsvField(fields, AzirLogDouble(fill_price, 5));
   AzirLogAddCsvField(fields, IntegerToString((int)duration_to_fill_seconds));
   AzirLogAddCsvField(fields, AzirLogDouble(mfe_points, 1));
   AzirLogAddCsvField(fields, AzirLogDouble(mae_points, 1));
   AzirLogAddCsvField(fields, exit_reason);
   AzirLogAddCsvField(fields, AzirLogDouble(gross_pnl, 2));
   AzirLogAddCsvField(fields, AzirLogDouble(net_pnl, 2));
   AzirLogAddCsvField(fields, AzirLogDouble(commission, 2));
   AzirLogAddCsvField(fields, AzirLogDouble(swap, 2));
   AzirLogAddCsvField(fields, AzirLogDouble(slippage_points, 1));
   AzirLogAddCsvField(fields, AzirLogBool(trailing_activated));
   AzirLogAddCsvField(fields, IntegerToString(trailing_modifications));
   AzirLogAddCsvField(fields, trailing_outcome);
   AzirLogAddCsvField(fields, AzirLogBool(opposite_order_cancelled));
   AzirLogAddCsvField(fields, notes);
   AzirLogWriteCsvLine(fields);
   FileFlush(g_azir_log_handle);
}

void AzirLogOpportunity(
   const datetime timestamp,
   const int day_of_week,
   const bool is_friday,
   const string timeframe,
   const int ny_open_hour,
   const int ny_open_minute,
   const int close_hour,
   const int swing_bars,
   const double lot_size,
   const int sl_points,
   const int tp_points,
   const int trailing_start_points,
   const int trailing_step_points,
   const double swing_high,
   const double swing_low,
   const double buy_entry,
   const double sell_entry,
   const double pending_distance_points,
   const double spread_points,
   const double ema20,
   const double prev_close,
   const double point,
   const double atr,
   const double atr_points,
   const bool atr_filter_enabled,
   const bool atr_filter_passed,
   const double atr_minimum,
   const double rsi,
   const bool rsi_gate_enabled,
   const bool rsi_gate_required,
   const double rsi_bullish_threshold,
   const double rsi_sell_threshold,
   const bool allow_buys,
   const bool allow_sells,
   const bool trend_filter_enabled,
   const bool buy_allowed_by_trend,
   const bool sell_allowed_by_trend,
   const bool buy_order_placed,
   const bool sell_order_placed,
   const uint buy_retcode,
   const uint sell_retcode,
   const string notes
)
{
   g_azir_setup_time   = timestamp;
   g_azir_buy_entry    = buy_entry;
   g_azir_sell_entry   = sell_entry;
   g_azir_log_event_id = TimeToString(timestamp, TIME_DATE) + "_" + g_azir_log_symbol + "_" + IntegerToString(g_azir_log_magic);
   double prev_vs_ema_points = (point > 0.0 ? (prev_close - ema20) / point : EMPTY_VALUE);
   AzirLogWriteRow(
      timestamp,
      "opportunity",
      day_of_week,
      is_friday,
      timeframe,
      ny_open_hour,
      ny_open_minute,
      close_hour,
      swing_bars,
      lot_size,
      sl_points,
      tp_points,
      trailing_start_points,
      trailing_step_points,
      swing_high,
      swing_low,
      buy_entry,
      sell_entry,
      pending_distance_points,
      spread_points,
      ema20,
      prev_close,
      prev_vs_ema_points,
      prev_close > ema20,
      atr,
      atr_points,
      atr_filter_enabled,
      atr_filter_passed,
      atr_minimum,
      rsi,
      rsi_gate_enabled,
      rsi_gate_required,
      true,
      rsi_bullish_threshold,
      rsi_sell_threshold,
      allow_buys,
      allow_sells,
      trend_filter_enabled,
      buy_allowed_by_trend,
      sell_allowed_by_trend,
      buy_order_placed,
      sell_order_placed,
      buy_retcode,
      sell_retcode,
      "",
      EMPTY_VALUE,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      "",
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      0,
      "",
      false,
      notes
   );
}

void AzirLogFridayBlocked(
   const datetime timestamp,
   const string timeframe,
   const int ny_open_hour,
   const int ny_open_minute,
   const int close_hour
)
{
   datetime day_open = iTime(g_azir_log_symbol, PERIOD_D1, 0);
   if(g_azir_last_friday_log_day == day_open)
      return;
   g_azir_last_friday_log_day = day_open;

   MqlDateTime dt;
   TimeToStruct(timestamp, dt);
   g_azir_log_event_id = TimeToString(timestamp, TIME_DATE) + "_" + g_azir_log_symbol + "_" + IntegerToString(g_azir_log_magic);
   AzirLogWriteRow(
      timestamp,
      "blocked_friday",
      dt.day_of_week,
      true,
      timeframe,
      ny_open_hour,
      ny_open_minute,
      close_hour,
      0,
      EMPTY_VALUE,
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      true,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      0,
      0,
      "",
      EMPTY_VALUE,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      "",
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      0,
      "",
      false,
      "NoTradeFridays blocked the daily opportunity before order evaluation."
   );
}

void AzirLogFill(
   const ulong deal_ticket,
   const long position_id,
   const long deal_type,
   const datetime fill_time,
   const double fill_price,
   const double point,
   const bool rsi_gate_enabled,
   const bool rsi_gate_required,
   const bool rsi_gate_passed,
   const double rsi_now
)
{
   g_azir_fill_time              = fill_time;
   g_azir_position_id            = position_id;
   g_azir_fill_price             = fill_price;
   g_azir_fill_side              = (deal_type == DEAL_TYPE_BUY ? 1 : -1);
   g_azir_mfe_points             = 0.0;
   g_azir_mae_points             = 0.0;
   g_azir_trailing_activated     = false;
   g_azir_trailing_modifications = 0;

   double expected_entry = (deal_type == DEAL_TYPE_BUY ? g_azir_buy_entry : g_azir_sell_entry);
   double slippage_points = EMPTY_VALUE;
   if(point > 0.0 && expected_entry > 0.0)
   {
      slippage_points = (deal_type == DEAL_TYPE_BUY)
                        ? (fill_price - expected_entry) / point
                        : (expected_entry - fill_price) / point;
   }
   long duration_seconds = (g_azir_setup_time > 0 ? (long)(fill_time - g_azir_setup_time) : 0);

   MqlDateTime dt;
   TimeToStruct(fill_time, dt);
   AzirLogWriteRow(
      fill_time,
      "fill",
      dt.day_of_week,
      dt.day_of_week == 5,
      "",
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      g_azir_buy_entry,
      g_azir_sell_entry,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      EMPTY_VALUE,
      rsi_now,
      rsi_gate_enabled,
      rsi_gate_required,
      rsi_gate_passed,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      0,
      0,
      AzirLogDealSide(deal_type),
      fill_price,
      duration_seconds,
      g_azir_mfe_points,
      g_azir_mae_points,
      "",
      EMPTY_VALUE,
      EMPTY_VALUE,
      HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION),
      HistoryDealGetDouble(deal_ticket, DEAL_SWAP),
      slippage_points,
      false,
      0,
      "",
      false,
      "Position fill detected from trade transaction."
   );
}

void AzirLogUpdateOpenPositionMetrics(const string symbol, const int magic, const double point)
{
   if(g_azir_fill_time <= 0 || point <= 0.0)
      return;

   for(int i=PositionsTotal()-1; i>=0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol)
         continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != magic)
         continue;

      bool is_buy = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_price = is_buy ? SymbolInfoDouble(symbol, SYMBOL_BID) : SymbolInfoDouble(symbol, SYMBOL_ASK);
      double favorable = is_buy ? (current_price - open_price) / point : (open_price - current_price) / point;
      double adverse = is_buy ? (open_price - current_price) / point : (current_price - open_price) / point;
      if(favorable > g_azir_mfe_points)
         g_azir_mfe_points = favorable;
      if(adverse > g_azir_mae_points)
         g_azir_mae_points = adverse;
   }
}

void AzirLogTrailingModified(const ulong ticket, const datetime timestamp, const double current_price, const double new_sl, const double profit_points)
{
   g_azir_trailing_activated = true;
   g_azir_trailing_modifications++;

   MqlDateTime dt;
   TimeToStruct(timestamp, dt);
   AzirLogWriteRow(
      timestamp,
      "trailing_modified",
      dt.day_of_week,
      dt.day_of_week == 5,
      "",
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      g_azir_buy_entry,
      g_azir_sell_entry,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      true,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      0,
      0,
      (g_azir_fill_side == 1 ? "buy" : "sell"),
      g_azir_fill_price,
      (g_azir_setup_time > 0 ? (long)(timestamp - g_azir_setup_time) : 0),
      g_azir_mfe_points,
      g_azir_mae_points,
      "",
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      true,
      g_azir_trailing_modifications,
      "sl_modified",
      g_azir_opposite_cancelled,
      "Trailing modified ticket=" + IntegerToString((int)ticket) +
      " current=" + DoubleToString(current_price, 5) +
      " new_sl=" + DoubleToString(new_sl, 5) +
      " profit_points=" + DoubleToString(profit_points, 1)
   );
}

void AzirLogOppositeOrderCancelled(const datetime timestamp, const ulong order_ticket)
{
   g_azir_opposite_cancelled = true;
   MqlDateTime dt;
   TimeToStruct(timestamp, dt);
   AzirLogWriteRow(
      timestamp,
      "opposite_pending_cancelled",
      dt.day_of_week,
      dt.day_of_week == 5,
      "",
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      g_azir_buy_entry,
      g_azir_sell_entry,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      true,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      0,
      0,
      (g_azir_fill_side == 1 ? "buy" : "sell"),
      g_azir_fill_price,
      (g_azir_setup_time > 0 ? (long)(timestamp - g_azir_setup_time) : 0),
      g_azir_mfe_points,
      g_azir_mae_points,
      "",
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      g_azir_trailing_activated,
      g_azir_trailing_modifications,
      "",
      true,
      "Cancelled opposite pending order ticket=" + IntegerToString((int)order_ticket)
   );
}

void AzirLogNoFillAtClose(const datetime timestamp, const int open_positions, const int pending_orders)
{
   datetime day_open = iTime(g_azir_log_symbol, PERIOD_D1, 0);
   if(g_azir_last_no_fill_log_day == day_open)
      return;
   if(g_azir_fill_time > 0 || pending_orders <= 0 || open_positions > 0)
      return;
   g_azir_last_no_fill_log_day = day_open;

   MqlDateTime dt;
   TimeToStruct(timestamp, dt);
   AzirLogWriteRow(
      timestamp,
      "no_fill_close_cleanup",
      dt.day_of_week,
      dt.day_of_week == 5,
      "",
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      g_azir_buy_entry,
      g_azir_sell_entry,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      true,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      0,
      0,
      "no_fill",
      EMPTY_VALUE,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      "close_hour_cleanup",
      0.0,
      0.0,
      0.0,
      0.0,
      EMPTY_VALUE,
      false,
      0,
      "",
      false,
      "Close hour reached with pending orders but no fill."
   );
}

void AzirLogExit(const ulong deal_ticket, const long position_id, const long deal_type, const long deal_reason, const datetime exit_time, const double exit_price)
{
   if(position_id <= 0)
      return;

   double gross_pnl = 0.0;
   double commission = 0.0;
   double swap = 0.0;
   HistorySelect(g_azir_setup_time > 0 ? g_azir_setup_time : 0, exit_time + 60);
   int total = HistoryDealsTotal();
   for(int i=0; i<total; ++i)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;
      if((long)HistoryDealGetInteger(ticket, DEAL_POSITION_ID) != position_id)
         continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != g_azir_log_symbol)
         continue;
      if((int)HistoryDealGetInteger(ticket, DEAL_MAGIC) != g_azir_log_magic)
         continue;
      gross_pnl += HistoryDealGetDouble(ticket, DEAL_PROFIT);
      commission += HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      swap += HistoryDealGetDouble(ticket, DEAL_SWAP);
   }
   double net_pnl = gross_pnl + commission + swap;
   string reason = AzirLogExitReason(deal_reason);
   string trailing_outcome = "";
   if(g_azir_trailing_activated)
      trailing_outcome = (reason == "take_profit" ? "tp_after_trailing" : "exit_after_trailing");

   MqlDateTime dt;
   TimeToStruct(exit_time, dt);
   AzirLogWriteRow(
      exit_time,
      "exit",
      dt.day_of_week,
      dt.day_of_week == 5,
      "",
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      0,
      0,
      0,
      0,
      EMPTY_VALUE,
      EMPTY_VALUE,
      g_azir_buy_entry,
      g_azir_sell_entry,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      true,
      EMPTY_VALUE,
      EMPTY_VALUE,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      0,
      0,
      (g_azir_fill_side == 1 ? "buy" : "sell"),
      g_azir_fill_price,
      (g_azir_setup_time > 0 ? (long)(g_azir_fill_time - g_azir_setup_time) : 0),
      g_azir_mfe_points,
      g_azir_mae_points,
      reason,
      gross_pnl,
      net_pnl,
      commission,
      swap,
      EMPTY_VALUE,
      g_azir_trailing_activated,
      g_azir_trailing_modifications,
      trailing_outcome,
      g_azir_opposite_cancelled,
      "Exit deal=" + IntegerToString((int)deal_ticket) + " exit_price=" + DoubleToString(exit_price, 5)
   );
}

#endif

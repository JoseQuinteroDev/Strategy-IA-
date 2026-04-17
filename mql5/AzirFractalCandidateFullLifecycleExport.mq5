//+------------------------------------------------------------------+
//|       Azir Fractal Candidate - Full Lifecycle Export EA          |
//|  Auxiliary EA for MT5 validation of swing_10_fractal candidate   |
//+------------------------------------------------------------------+
#property copyright   "© 2025, Jose Quintero"
#property link        "https://t.me/JoseTrader10"
#property version     "2.20"
#property description "Auxiliary Azir lifecycle exporter using swing_10_fractal. Does not replace the official Azir EA."
#property strict

#include <Trade/Trade.mqh>
#include "AzirEventLogger.mqh"

CTrade trade;

//==================================================================//
//                         INPUTS (GROUPED)                         //
//==================================================================//

input group "Trade Size & Risk"
input double         LotSize                 = 0.10;      // Position size (lots)
input int            SL_Points               = 500;       // Stop Loss in points (multiplied by Symbol POINT)
input int            TP_Points               = 500;       // Take Profit in points (multiplied by Symbol POINT)
input int            Trailing_Start_Points   = 90;        // Activate trailing at this profit (points)
input int            Trailing_Step_Points    = 50;        // Trailing step (points)
input int            MagicNumber             = 123456321; // Magic number

input group "Session & Structure"
input ENUM_TIMEFRAMES Timeframe              = PERIOD_M5; // EMA & swing timeframe
input int            NY_Open_Hour            = 16;        // New York open hour (broker/server time, GMT+3)
input int            NY_Open_Minute          = 30;        // New York open minute (broker/server time, GMT+3)
input int            Close_Hour              = 22;        // Hour to close positions & delete pendings (broker/server time, GMT+3)
input int            SwingBars               = 10;        // Bars to find swing highs/lows (on 'Timeframe')
input int            FractalSideBars         = 2;         // Confirmed pivot side bars for swing_10_fractal

input group "Directional Controls"
input bool           AllowBuys               = true;      // Allow Buy trades
input bool           AllowSells              = true;      // Allow Sell trades
input bool           AllowTrendFilter        = true;      // EMA trend filter for entries

input group "Volatility Filter (ATR)"
input bool              AllowAtrFilter       = true;      // Enable ATR minimum filter
input int               ATR_Period           = 14;        // ATR period
input ENUM_TIMEFRAMES   ATR_Timeframe        = PERIOD_M5; // ATR timeframe
input double            ATR_Minimum          = 100;       // Minimum ATR threshold (points)

input group "RSI Distance Gate"
input bool              AllowRsiFilter               = true;      // Enable RSI gate when distance condition is met
input int               RSI_Period                   = 14;        // RSI period
input ENUM_TIMEFRAMES   RSI_Timeframe                = PERIOD_M1; // RSI timeframe (default M1)
input double            RSI_Bullish_Threshold        = 70.0;      // BUY requires RSI >= this
input double            RSI_Sell_Threshold           = 30.0;      // SELL requires RSI <= this
input double            MinDistanceBetweenPendings   = 200.0;     // Min distance between BuyStop & SellStop to require RSI (in points)

input group "Other"
input bool           NoTradeFridays          = true;      // Don't trade on Fridays

//=========================== UI & Display ==========================//
input group "UI & Display"
input bool           EnableGraphics          = true;      // Master ON/OFF for all on-chart visuals
input bool           ShowHUD                 = true;      // Show on-chart HUD
input int            HUD_Refresh_Seconds     = 1;         // Refresh interval (seconds)
input bool           HUD_ShowEAOnlyPnL       = true;      // Show EA P/L (by symbol+magic)
input bool           HUD_ShowAccountStats    = true;      // Show account stats

input group "Event Logging"
input bool           EnableEventLogging      = true;      // Export Azir decision/fill/exit events to CSV
input string         EventLogFileName        = "fractal_candidate_full_lifecycle_event_log.csv";
input bool           EventLogUseCommonFolder = true;      // true = common MetaTrader Files folder


//==================================================================//
//                         GLOBAL / HANDLES                         //
//==================================================================//

static bool     orders_placed_today     = false;
static datetime current_day;
int             ema_handle;
int             atr_handle;
int             rsi_handle;

bool   rsi_gate_required_today = false;
double last_buy_entry_price    = 0.0;
double last_sell_entry_price   = 0.0;

/*#define ALLOWED_BROKER  "FundedNext Ltd"  // EXACT broker company name
#define ALLOWED_ACCOUNT 11452753     */     // Allowed account number

//------------------------ HUD helpers (NEW) ------------------------//
string TFToString(const ENUM_TIMEFRAMES tf)
{
   switch(tf)
   {
      case PERIOD_M1:   return "M1";
      case PERIOD_M2:   return "M2";
      case PERIOD_M3:   return "M3";
      case PERIOD_M4:   return "M4";
      case PERIOD_M5:   return "M5";
      case PERIOD_M6:   return "M6";
      case PERIOD_M10:  return "M10";
      case PERIOD_M12:  return "M12";
      case PERIOD_M15:  return "M15";
      case PERIOD_M20:  return "M20";
      case PERIOD_M30:  return "M30";
      case PERIOD_H1:   return "H1";
      case PERIOD_H2:   return "H2";
      case PERIOD_H3:   return "H3";
      case PERIOD_H4:   return "H4";
      case PERIOD_H6:   return "H6";
      case PERIOD_H8:   return "H8";
      case PERIOD_H12:  return "H12";
      case PERIOD_D1:   return "D1";
      case PERIOD_W1:   return "W1";
      case PERIOD_MN1:  return "MN1";
      default:          return IntegerToString((int)tf);
   }
}

bool FindLastConfirmedFractalHigh(
   const string symbol,
   const ENUM_TIMEFRAMES timeframe,
   const int lookback,
   const int side,
   double &value
)
{
   if(lookback < side * 2 + 1 || side < 1)
      return false;

   const int first_shift = side + 1;      // newest fully-confirmed closed pivot
   const int last_shift  = lookback - side;
   if(last_shift < first_shift)
      return false;

   for(int shift = first_shift; shift <= last_shift; shift++)
   {
      const double candidate = iHigh(symbol, timeframe, shift);
      bool is_pivot = true;

      for(int offset = 1; offset <= side; offset++)
      {
         if(candidate <= iHigh(symbol, timeframe, shift - offset) ||
            candidate <= iHigh(symbol, timeframe, shift + offset))
         {
            is_pivot = false;
            break;
         }
      }

      if(is_pivot)
      {
         value = candidate;
         return true;
      }
   }

   return false;
}

bool FindLastConfirmedFractalLow(
   const string symbol,
   const ENUM_TIMEFRAMES timeframe,
   const int lookback,
   const int side,
   double &value
)
{
   if(lookback < side * 2 + 1 || side < 1)
      return false;

   const int first_shift = side + 1;      // newest fully-confirmed closed pivot
   const int last_shift  = lookback - side;
   if(last_shift < first_shift)
      return false;

   for(int shift = first_shift; shift <= last_shift; shift++)
   {
      const double candidate = iLow(symbol, timeframe, shift);
      bool is_pivot = true;

      for(int offset = 1; offset <= side; offset++)
      {
         if(candidate >= iLow(symbol, timeframe, shift - offset) ||
            candidate >= iLow(symbol, timeframe, shift + offset))
         {
            is_pivot = false;
            break;
         }
      }

      if(is_pivot)
      {
         value = candidate;
         return true;
      }
   }

   return false;
}

bool ComputeFractalSwingHigh(
   const string symbol,
   const ENUM_TIMEFRAMES timeframe,
   const int lookback,
   const int side,
   double &value
)
{
   if(FindLastConfirmedFractalHigh(symbol, timeframe, lookback, side, value))
      return true;

   const int rolling_shift = iHighest(symbol, timeframe, MODE_HIGH, lookback, 1);
   if(rolling_shift < 0)
      return false;

   value = iHigh(symbol, timeframe, rolling_shift);
   return true;
}

bool ComputeFractalSwingLow(
   const string symbol,
   const ENUM_TIMEFRAMES timeframe,
   const int lookback,
   const int side,
   double &value
)
{
   if(FindLastConfirmedFractalLow(symbol, timeframe, lookback, side, value))
      return true;

   const int rolling_shift = iLowest(symbol, timeframe, MODE_LOW, lookback, 1);
   if(rolling_shift < 0)
      return false;

   value = iLow(symbol, timeframe, rolling_shift);
   return true;
}

double GetEA_FloatingPnL()
{
   double sum = 0.0;
   for(int i=PositionsTotal()-1; i>=0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=MagicNumber) continue;

      // POSITION_PROFIT in MT5 is NET of swap & commission.
      sum += PositionGetDouble(POSITION_PROFIT);
   }
   return sum;
}


double GetEA_RealizedPnL_Today()
{
   // Use the current trading day open (server time)
   datetime day_open = current_day; // set/updated elsewhere
   if(day_open==0) day_open = iTime(_Symbol, PERIOD_D1, 0);

   HistorySelect(day_open, TimeCurrent());

   double sum = 0.0;
   int total = HistoryDealsTotal();
   for(int i=0; i<total; ++i)
   {
      ulong deal = HistoryDealGetTicket(i);
      if(deal==0) continue;

      if(HistoryDealGetString(deal, DEAL_SYMBOL) != _Symbol) continue;
      if((long)HistoryDealGetInteger(deal, DEAL_MAGIC) != MagicNumber) continue;

      long entry = (long)HistoryDealGetInteger(deal, DEAL_ENTRY);
      // Count realized on exits (and possible in/out netting)
      bool is_exit = (entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_OUT_BY || entry==DEAL_ENTRY_INOUT);
      if(!is_exit) continue;

      sum += HistoryDealGetDouble(deal, DEAL_PROFIT)
           + HistoryDealGetDouble(deal, DEAL_SWAP)
           + HistoryDealGetDouble(deal, DEAL_COMMISSION);
   }
   return sum;
}

int CountEA_OpenPositions()
{
   int cnt=0;
   for(int i=PositionsTotal()-1; i>=0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=MagicNumber) continue;
      ++cnt;
   }
   return cnt;
}

int CountEA_PendingOrders()
{
   int cnt=0;
   for(int i=OrdersTotal()-1; i>=0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(!OrderSelect(ticket)) continue;
      if(OrderGetString(ORDER_SYMBOL)!=_Symbol) continue;
      if(OrderGetInteger(ORDER_MAGIC)!=MagicNumber) continue;
      ++cnt;
   }
   return cnt;
}

void UpdateHUD()
{
   if(!ShowHUD)
   {
      Comment("");
      return;
   }

   string title = "AzirIA MT5  v" + (string)__DATE__ + "  (ver 2.2)";
   // Using build date to give a hint of when this was compiled; label shows ver 2.2

   string broker = AccountInfoString(ACCOUNT_COMPANY);
   long   login  = (long)AccountInfoInteger(ACCOUNT_LOGIN);
   string sym    = _Symbol;
   string tf     = TFToString(Timeframe);
   string now    = TimeToString(TimeCurrent(), TIME_SECONDS);

   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);

   double ea_float   = HUD_ShowEAOnlyPnL ? GetEA_FloatingPnL() : 0.0;
   double ea_real    = HUD_ShowEAOnlyPnL ? GetEA_RealizedPnL_Today() : 0.0;
   int    ea_pos     = CountEA_OpenPositions();
   int    ea_pend    = CountEA_PendingOrders();

   string acc_stats = HUD_ShowAccountStats
      ? StringFormat("Balance: %.2f   Equity: %.2f   FreeMargin: %.2f", balance, equity, freeMargin)
      : "";

   string ea_stats = StringFormat("EA Floating P/L: %.2f   EA Realized Today: %.2f   Open: %d   Pendings: %d",
                                  ea_float, ea_real, ea_pos, ea_pend);

   string status_rsi = StringFormat("RSI Gate: %s   (MinDist: %.1f pts)",
                        (rsi_gate_required_today ? "ACTIVE" : "idle"), MinDistanceBetweenPendings);

   string header = "Azir fractal lifecycle export (v2.2-fractal)  |  Broker: " + broker + "  |  Account: " + IntegerToString(login);
   string line2  = "Symbol: " + sym + "  TF: " + tf + "  |  Server Time: " + now;

   string out;
   if(HUD_ShowAccountStats)
      out = header + "\n" + line2 + "\n" + acc_stats + "\n" + ea_stats + "\n" + status_rsi;
   else
      out = header + "\n" + line2 + "\n" + ea_stats + "\n" + status_rsi;

   Comment(out);
}

//==================================================================//
//                              INIT                                 //
//==================================================================//
int OnInit()
{
   /*string broker_now = AccountInfoString(ACCOUNT_COMPANY);
   long   account_now = AccountInfoInteger(ACCOUNT_LOGIN);

   if(broker_now != ALLOWED_BROKER || account_now != ALLOWED_ACCOUNT)
   {
      string msg = "EA authorized only for:\n"
                   "Broker: " + ALLOWED_BROKER + "\n"
                   "Account: " + IntegerToString(ALLOWED_ACCOUNT) + "\n\n"
                   "Current:\n"
                   "Broker: " + broker_now + "\n"
                   "Account: " + IntegerToString(account_now);

      Print(msg);
      MessageBox(msg, "Azir MT5 - License", MB_ICONERROR);
      return(INIT_FAILED);
   }*/

   current_day = iTime(_Symbol, PERIOD_D1, 0);
   if(SwingBars < FractalSideBars * 2 + 1)
   {
      Print("Error: SwingBars must be at least 2 * FractalSideBars + 1 for swing_10_fractal.");
      return(INIT_FAILED);
   }

   trade.SetExpertMagicNumber(MagicNumber);
   AzirLogConfigure(EnableEventLogging, EventLogFileName, EventLogUseCommonFolder);
   AzirLogInit(_Symbol, MagicNumber);

   ema_handle = iMA(_Symbol, Timeframe, 20, 0, MODE_EMA, PRICE_CLOSE);
   if(ema_handle == INVALID_HANDLE){
      Print("Error creating EMA handle");
      return(INIT_FAILED);
   }

   atr_handle = iATR(_Symbol, ATR_Timeframe, ATR_Period);
   if(atr_handle == INVALID_HANDLE){
      Print("Error creating ATR handle");
      return(INIT_FAILED);
   }

   rsi_handle = iRSI(_Symbol, RSI_Timeframe, RSI_Period, PRICE_CLOSE);
   if(rsi_handle == INVALID_HANDLE){
      Print("Error creating RSI handle");
      return(INIT_FAILED);
   }

   //--- HUD timer (NEW)
   if(ShowHUD && HUD_Refresh_Seconds > 0)
      EventSetTimer(HUD_Refresh_Seconds);

   UpdateHUD(); // initial draw

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(ema_handle!=INVALID_HANDLE) IndicatorRelease(ema_handle);
   if(atr_handle!=INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(rsi_handle!=INVALID_HANDLE) IndicatorRelease(rsi_handle);

   AzirLogClose();

   //--- kill timer + clear HUD (NEW)
   EventKillTimer();
   Comment("");
}

//==================================================================//
//                              TICK                                 //
//==================================================================//
void OnTick()
{
   // Daily reset
   datetime now_time = TimeCurrent();
   datetime new_day  = iTime(_Symbol, PERIOD_D1, 0);
   if(new_day != current_day){
      orders_placed_today     = false;
      rsi_gate_required_today = false;
      last_buy_entry_price    = 0.0;
      last_sell_entry_price   = 0.0;
      current_day             = new_day;
      AzirLogResetDailyState();
   }

   // Friday filter
   MqlDateTime dt; TimeToStruct(now_time, dt);
   bool is_friday = (dt.day_of_week == 5);
   if (NoTradeFridays && is_friday){
      if(dt.hour == NY_Open_Hour && dt.min == NY_Open_Minute)
         AzirLogFridayBlocked(now_time, TFToString(Timeframe), NY_Open_Hour, NY_Open_Minute, Close_Hour);
      Print("We don't trade on Fridays");
      UpdateHUD();
      return;
   }

   // Place pendings at session open (exact minute)
   if(dt.hour == NY_Open_Hour && dt.min == NY_Open_Minute && !orders_placed_today)
   {
      // EMA (closed bar)
      double ema_arr[];
      if(CopyBuffer(ema_handle, 0, 1, 1, ema_arr) <= 0) return;
      double ema_val = ema_arr[0];

      // Basics
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double spread_points = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);

      // Swings: only the structural swing definition differs from official Azir.
      int    bars       = SwingBars;
      double swing_hi   = 0.0;
      double swing_lo   = 0.0;
      if(!ComputeFractalSwingHigh(_Symbol, Timeframe, bars, FractalSideBars, swing_hi))
      {
         Print("Could not compute swing_10_fractal high.");
         UpdateHUD();
         return;
      }
      if(!ComputeFractalSwingLow(_Symbol, Timeframe, bars, FractalSideBars, swing_lo))
      {
         Print("Could not compute swing_10_fractal low.");
         UpdateHUD();
         return;
      }
      double prev_close = iClose(_Symbol, Timeframe, 1);

      // Entry offset (in points). This value is intentionally logged as-is;
      // it is part of the current Azir implementation and is not changed here.
      double offset_points_value = 5;                // 5 points (e.g., 0.05 on XAUUSD if POINT=0.01)
      double entry_offset        = offset_points_value * point;

      double buy_entry  = swing_hi + entry_offset;
      double sell_entry = swing_lo - entry_offset;
      double pending_distance_points = (point > 0.0 ? (buy_entry - sell_entry) / point : 0.0);
      bool buy_allowed_by_trend  = (!AllowTrendFilter || prev_close > ema_val) && AllowBuys;
      bool sell_allowed_by_trend = (!AllowTrendFilter || prev_close < ema_val) && AllowSells;

      // ATR (closed bar)
      double atr_vals[];
      if(CopyBuffer(atr_handle, 0, 1, 1, atr_vals) <= 0){
         Print("Couldn't get ATR (closed bar).");
         UpdateHUD();
         return;
      }
      double atr_points = atr_vals[0] / point; // ATR in points
      double rsi_setup = EMPTY_VALUE;
      double rsi_setup_arr[];
      if(CopyBuffer(rsi_handle, 0, 1, 1, rsi_setup_arr) > 0)
         rsi_setup = rsi_setup_arr[0];

      if(AllowAtrFilter && atr_points < ATR_Minimum){
         Print("ATR (", DoubleToString(atr_points,1), ") is below minimum (", DoubleToString(ATR_Minimum,1), "). No trading today.");
         AzirLogOpportunity(now_time, dt.day_of_week, is_friday, TFToString(Timeframe),
                            NY_Open_Hour, NY_Open_Minute, Close_Hour, SwingBars,
                            LotSize, SL_Points, TP_Points,
                            Trailing_Start_Points, Trailing_Step_Points,
                            swing_hi, swing_lo, buy_entry, sell_entry,
                            pending_distance_points, spread_points, ema_val, prev_close,
                            point, atr_vals[0], atr_points, AllowAtrFilter, false,
                            ATR_Minimum, rsi_setup, AllowRsiFilter, false,
                            RSI_Bullish_Threshold, RSI_Sell_Threshold, AllowBuys, AllowSells,
                            AllowTrendFilter, buy_allowed_by_trend, sell_allowed_by_trend,
                            false, false, 0, 0, "ATR filter failed; no orders were sent.");
         UpdateHUD();
         return;
      }

      bool buyPlaced  = false;
      bool sellPlaced = false;
      uint buyRetcode = 0;
      uint sellRetcode = 0;

      // Place with/without trend filter
      if(AllowTrendFilter){
         if(prev_close > ema_val && AllowBuys){
            buyPlaced = trade.BuyStop(LotSize, buy_entry, _Symbol,
                                      buy_entry - SL_Points * point,
                                      buy_entry + TP_Points * point,
                                      ORDER_TIME_GTC, 0, "Azir BuyStop");
            buyRetcode = trade.ResultRetcode();
            if(!buyPlaced) Print("BuyStop failed. Err=", GetLastError());
         }
         else if(prev_close < ema_val && AllowSells){
            sellPlaced = trade.SellStop(LotSize, sell_entry, _Symbol,
                                        sell_entry + SL_Points * point,
                                        sell_entry - TP_Points * point,
                                        ORDER_TIME_GTC, 0, "Azir SellStop");
            sellRetcode = trade.ResultRetcode();
            if(!sellPlaced) Print("SellStop failed. Err=", GetLastError());
         }
      } else {
         if(AllowBuys){
            buyPlaced = trade.BuyStop(LotSize, buy_entry, _Symbol,
                                      buy_entry - SL_Points * point,
                                      buy_entry + TP_Points * point,
                                      ORDER_TIME_GTC, 0, "Azir BuyStop");
            buyRetcode = trade.ResultRetcode();
            if(!buyPlaced) Print("BuyStop failed. Err=", GetLastError());
         }
         if(AllowSells){
            sellPlaced = trade.SellStop(LotSize, sell_entry, _Symbol,
                                        sell_entry + SL_Points * point,
                                        sell_entry - TP_Points * point,
                                        ORDER_TIME_GTC, 0, "Azir SellStop");
            sellRetcode = trade.ResultRetcode();
            if(!sellPlaced) Print("SellStop failed. Err=", GetLastError());
         }
      }

      // Activate RSI gate only if:
      // 1) Allowed, 2) BOTH pendings placed, 3) Distance >= threshold (in points)
      rsi_gate_required_today = false;
      if(AllowRsiFilter && buyPlaced && sellPlaced){
         double point2 = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         double distance_points = (buy_entry - sell_entry) / point2; // >0
         if(distance_points >= MinDistanceBetweenPendings){
            rsi_gate_required_today = true;
            last_buy_entry_price  = buy_entry;
            last_sell_entry_price = sell_entry;
            Print("RSI gate ACTIVATED. Distance=", DoubleToString(distance_points,1), " points.");
         } else {
            Print("RSI gate NOT activated. Distance=", DoubleToString(distance_points,1),
                 " < threshold ", MinDistanceBetweenPendings);
         }
      }

      // Mark the day only if at least one order was indeed placed
      string setup_notes = "";
      if(!buyPlaced && !sellPlaced)
         setup_notes = "Fractal lifecycle candidate: no pending order placed after trend/direction/order-send evaluation.";
      else if(rsi_gate_required_today)
         setup_notes = "Fractal lifecycle candidate: pending order placement evaluated; RSI gate activated because both sides were placed and distance threshold passed.";
      else
         setup_notes = "Fractal lifecycle candidate: pending order placement evaluated; RSI gate inactive for this opportunity.";
      AzirLogOpportunity(now_time, dt.day_of_week, is_friday, TFToString(Timeframe),
                         NY_Open_Hour, NY_Open_Minute, Close_Hour, SwingBars,
                         LotSize, SL_Points, TP_Points,
                         Trailing_Start_Points, Trailing_Step_Points,
                         swing_hi, swing_lo, buy_entry, sell_entry,
                         pending_distance_points, spread_points, ema_val, prev_close,
                         point, atr_vals[0], atr_points, AllowAtrFilter, true,
                         ATR_Minimum, rsi_setup, AllowRsiFilter, rsi_gate_required_today,
                         RSI_Bullish_Threshold, RSI_Sell_Threshold, AllowBuys, AllowSells,
                         AllowTrendFilter, buy_allowed_by_trend, sell_allowed_by_trend,
                         buyPlaced, sellPlaced, buyRetcode, sellRetcode, setup_notes);

      orders_placed_today = (buyPlaced || sellPlaced);
   }

   // Closing time: close positions & delete pendings (symbol + magic only)
   if(dt.hour == Close_Hour && dt.min == 0)
   {
      AzirLogNoFillAtClose(now_time, CountEA_OpenPositions(), CountEA_PendingOrders());

      // Close positions
      for(int i=PositionsTotal()-1; i>=0; i--){
         ulong pos_ticket = PositionGetTicket(i);
         if(PositionSelectByTicket(pos_ticket)
            && PositionGetString(POSITION_SYMBOL)==_Symbol
            && PositionGetInteger(POSITION_MAGIC)==MagicNumber)
         {
            trade.PositionClose(pos_ticket);
         }
      }
      // Delete pending orders
      for(int i=OrdersTotal()-1; i>=0; i--){
         ulong ord_ticket = OrderGetTicket(i);
         if(OrderSelect(ord_ticket)
            && OrderGetString(ORDER_SYMBOL)==_Symbol
            && OrderGetInteger(ORDER_MAGIC)==MagicNumber)
         {
            trade.OrderDelete(ord_ticket);
         }
      }
   }

   // Trailing stop (fixed step)
   int    stopLevelPts = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double stopLevel    = stopLevelPts * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double point        = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   AzirLogUpdateOpenPositionMetrics(_Symbol, MagicNumber, point);

   for(int i=PositionsTotal()-1; i>=0; i--){
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)
         || PositionGetString(POSITION_SYMBOL)!=_Symbol
         || PositionGetInteger(POSITION_MAGIC)!=MagicNumber)
         continue;

      bool   isBuy      = PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
      double openPrice  = PositionGetDouble(POSITION_PRICE_OPEN);
      double currPrice  = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double slOld      = PositionGetDouble(POSITION_SL);
      double profitPts  = (isBuy ? (currPrice-openPrice) : (openPrice-currPrice)) / point;

      if(profitPts >= Trailing_Start_Points){
         double newSl;
         if(isBuy) {
            newSl = currPrice - Trailing_Step_Points * point;
            if((slOld==0.0 || newSl > slOld) && (currPrice - newSl) > stopLevel)
            {
               double normalized_sl = NormalizeDouble(newSl, _Digits);
               if(trade.PositionModify(ticket, normalized_sl, PositionGetDouble(POSITION_TP)))
                  AzirLogTrailingModified(ticket, now_time, currPrice, normalized_sl, profitPts);
            }
         } else {
            newSl = currPrice + Trailing_Step_Points * point;
            if((slOld==0.0 || newSl < slOld) && (newSl - currPrice) > stopLevel)
            {
               double normalized_sl = NormalizeDouble(newSl, _Digits);
               if(trade.PositionModify(ticket, normalized_sl, PositionGetDouble(POSITION_TP)))
                  AzirLogTrailingModified(ticket, now_time, currPrice, normalized_sl, profitPts);
            }
         }
      }
   }

   //--- HUD refresh on tick (NEW)
   UpdateHUD();
}

//==================================================================//
//                         TIMER (HUD refresh)                       //
//==================================================================//
void OnTimer()
{
   UpdateHUD();
}

//==================================================================//
//                    TRADE TRANSACTIONS (RSI GATE)                  //
//==================================================================//
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult&  result)
{
   // Only when a deal is added (execution)
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   ulong deal_ticket = trans.deal;
   if(!HistoryDealSelect(deal_ticket))
      return;

   string deal_symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   long   deal_magic  = (long)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
   long   deal_type   = (long)HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
   long   pos_id      = (long)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);

   if(deal_symbol != _Symbol || deal_magic != MagicNumber)
      return;

   long     deal_entry  = (long)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
   long     deal_reason = (long)HistoryDealGetInteger(deal_ticket, DEAL_REASON);
   datetime deal_time   = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
   double   deal_price  = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
   double   point       = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   if(deal_entry == DEAL_ENTRY_OUT || deal_entry == DEAL_ENTRY_OUT_BY)
      AzirLogExit(deal_ticket, pos_id, deal_type, deal_reason, deal_time, deal_price);

   if(deal_entry == DEAL_ENTRY_IN || deal_entry == DEAL_ENTRY_INOUT)
   {
      double rsi_now_for_log = EMPTY_VALUE;
      double rsi_val_for_log[];
      if(CopyBuffer(rsi_handle, 0, 0, 1, rsi_val_for_log) > 0)
         rsi_now_for_log = rsi_val_for_log[0];

      bool rsi_pass_for_log = true;
      if(AllowRsiFilter && rsi_gate_required_today)
      {
         if(deal_type == DEAL_TYPE_BUY)
            rsi_pass_for_log = (rsi_now_for_log >= RSI_Bullish_Threshold);
         else if(deal_type == DEAL_TYPE_SELL)
            rsi_pass_for_log = (rsi_now_for_log <= RSI_Sell_Threshold);
      }

      AzirLogFill(deal_ticket, pos_id, deal_type, deal_time, deal_price, point,
                  AllowRsiFilter, rsi_gate_required_today, rsi_pass_for_log, rsi_now_for_log);
   }

   // Enforce RSI gate only if active today
   if(!AllowRsiFilter || !rsi_gate_required_today)
      return;

   // Read current RSI
   double rsi_val_arr[];
   if(CopyBuffer(rsi_handle, 0, 0, 1, rsi_val_arr) <= 0){
      Print("Couldn't read RSI in OnTradeTransaction. Gate not applied.");
      return;
   }
   double rsi_now = rsi_val_arr[0];

   bool pass = true;
   if(deal_type == DEAL_TYPE_BUY){
      pass = (rsi_now >= RSI_Bullish_Threshold);
   } else if(deal_type == DEAL_TYPE_SELL){
      pass = (rsi_now <= RSI_Sell_Threshold);
   } else {
      return; // not a buy/sell open
   }

   if(!pass){
      Print("RSI gate blocks the entry. RSI=", DoubleToString(rsi_now,1),
            " | type=", (deal_type==DEAL_TYPE_BUY ? "BUY" : "SELL"));
      if(pos_id > 0){
         trade.PositionClose((ulong)pos_id);
      } else {
         // Fallback: close by symbol+magic (netting mode safety)
         for(int i=PositionsTotal()-1; i>=0; i--){
            ulong pticket = PositionGetTicket(i);
            if(PositionSelectByTicket(pticket)
               && PositionGetString(POSITION_SYMBOL)==_Symbol
               && PositionGetInteger(POSITION_MAGIC)==MagicNumber)
            {
               trade.PositionClose(pticket);
            }
         }
      }
   } else {
      Print("RSI gate OK. RSI=", DoubleToString(rsi_now,1),
            " | ", (deal_type==DEAL_TYPE_BUY ? "BUY" : "SELL"));
      // Optional: cancel the opposite pending to avoid dual fills
      for(int i=OrdersTotal()-1; i>=0; i--){
         ulong oticket = OrderGetTicket(i);
         if(OrderSelect(oticket)
            && OrderGetString(ORDER_SYMBOL)==_Symbol
            && OrderGetInteger(ORDER_MAGIC)==MagicNumber)
         {
            ENUM_ORDER_TYPE otype = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
            if( (deal_type==DEAL_TYPE_BUY  && otype==ORDER_TYPE_SELL_STOP) ||
                (deal_type==DEAL_TYPE_SELL && otype==ORDER_TYPE_BUY_STOP) )
            {
               if(trade.OrderDelete(oticket))
                  AzirLogOppositeOrderCancelled(TimeCurrent(), oticket);
            }
         }
      }
   }

   // Update HUD after fills
   UpdateHUD();
}

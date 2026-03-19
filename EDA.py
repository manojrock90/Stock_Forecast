class Report:
    def __init__(self,df, price_type,SMA_period = None, EMA_period = None, volatility_period = None, rsi_period = None):
        self.df = df.copy()
        self.price_type = price_type
        self.SMA_period = SMA_period
        self.EMA_period = EMA_period
        self.volatility_period = volatility_period
        self.rsi_period = rsi_period
        
    def technical_indicators(self):
        # Simple Moving Average
        if self.SMA_period:
            self.df[f'{self.price_type}_SMA_{self.SMA_period}'] = (
                self.df[self.price_type].rolling(self.SMA_period).mean()
            )

        # Exponential Moving Average
        if self.EMA_period:
            self.df[f'{self.price_type}_EMA_{self.EMA_period}'] = (
                self.df[self.price_type].ewm(span=self.EMA_period, adjust=False).mean()
            )

        # Returns
        self.df['Returns'] = self.df[self.price_type].pct_change()

        # Volatility
        if self.volatility_period:
            self.df[f'Volatility_{self.volatility_period}'] = (
                self.df['Returns'].rolling(self.volatility_period).std()
            )
        self.df['Day'] = self.df['Datetime'].dt.day_name()
        self.df.groupby('Day')['Returns'].mean()    
    #Bollinger Bands and RSI
        delta = self.df[self.price_type].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta<0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100/(1 + rs))

        self.df['Upper_Band'] = self.df[f'{self.price_type}_SMA_{self.SMA_period}'] + 2 * self.df[self.price_type].rolling(self.SMA_period).std()
        self.df['Lower_Band'] = self.df[f'{self.price_type}_SMA_{self.SMA_period}'] - 2 * self.df[self.price_type].rolling(self.SMA_period).std()

        return self.df



import numpy as np
import math

def MACD(DF, a, b, c):
    intra_price = DF.copy()
    intra_price["MA_FAST"] = intra_price["Close"].ewm(span = a, min_periods = a).mean()
    intra_price["MA_SLOW"] = intra_price["Close"].ewm(span = b, min_periods = b).mean()
    intra_price["MACD"] = intra_price["MA_FAST"] - intra_price["MA_SLOW"]
    intra_price["SIGNAL"] = intra_price["Close"].ewm(span = c, min_periods = c).mean()
    intra_price.dropna(inplace=True)
    return intra_price

def SMA(DF, day):
    copy = DF.copy()
    sma = copy.rolling(window=day, min_periods=day).mean() 
    return sma

def EMA(DF, day):
    copy = DF.copy()
    sma = copy.ewm(span=day, min_periods=day).mean() 
    return sma

def ATR(DF, n):
    df = DF.copy()
    df["H-L"] = abs(df["High"] - df["Low"]).shift(1)
    df["H-PC"] = abs(df["High"] - df["Close"]).shift(1)
    df["L-PC"] = abs(df["Low"] - df["Close"]).shift(1)
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].rolling(n).mean()
    df2 = df.drop(["H-L", "H-PC", "L-PC"], axis=1)
    return df2

def BollBnd(DF, n):
    df = DF.copy()
    df["MA"] = df["Close"].rolling(n).mean()
    df["BB_UP"] = df["MA"] + 2*df["MA"].rolling(n).std()
    df["BB_DOWN"] = df["MA"] - 2*df["MA"].rolling(n).std()
    df["BB_RANGE"] = df["BB_UP"] - df["BB_DOWN"]
    df.dropna(inplace=True)
    df2 = df.drop(["BB_UP", "BB_DOWN"], axis=1)
    return df

def RSI(DF, n):
    df = DF.copy()
    print(len(df))
    df["delta"] = df["Close"] - df["Close"].shift(1)
    df["gain"] = np.where(df["delta"]>=0, df["delta"], 0)
    df["loss"] = np.where(df["delta"]<0, abs(df["delta"]), 0)
    avg_gain = []
    avg_loss = []
    gain = df["gain"].tolist()
    loss = df["loss"].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df["gain"].rolling(n).mean().tolist()[n])
            avg_loss.append(df["loss"].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append((avg_gain[i - 1]*(n - 1) + gain[i])/n)
            avg_loss.append((avg_loss[i - 1]*(n - 1) + loss[i])/n)
    df["avg_gain"] = np.array(avg_gain)

    df["avg_loss"] = np.array(avg_loss)
    df["RS"] = df["avg_gain"]/df["avg_loss"]
    df["RSI"] = 100 - (100/(1+df["RS"]))
    return df["RSI"]

def ADX(DF, n):
    df2 = DF.copy()
    df2["TR"] = ATR(df2, n)["TR"]
    df2["DMplus"] = np.where((df2["High"] - df2["High"].shift(1))>(df2["Low"].shift(1)-df2["Low"]), 
                             df2["High"] - df2["High"].shift(1),
                             0)
    df2["DMplus"] = np.where(df2["DMplus"]<0, 0, df2["DMplus"])
    df2["DMminus"] = np.where((df2["Low"].shift(1)-df2["Low"])>(df2["High"] - df2["High"].shift(1)),
                             df2["Low"].shift(1)-df2["Low"],
                             0)
    df2["DMminus"] = np.where(df2["DMminus"]<0, 0, df2["DMminus"])
    TRn=[]
    DMplusN=[]
    DMminusN=[]
    TR = df2["TR"].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i<n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2["TR"].rolling(n).sum().tolist()[n])
            DMplusN.append(df2["DMplus"].rolling(n).sum().tolist()[n])
            DMminusN.append(df2["DMminus"].rolling(n).sum().tolist()[n])
        else:
            TRn.append(TRn[i-1] - (TRn[i-1]/14) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/14) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/14) + DMminus[i])
            
    df2["TRn"] = np.array(TRn)
    df2["DMplusN"] = np.array(DMplusN)
    df2["DMminusN"] = np.array(DMminusN)
    df2["DIplusN"] = (100*df2["DMplusN"]/df2["TRn"])
    df2["DIminusN"] = (100*df2["DMminusN"]/df2["TRn"])
    df2["DIsum"] =  df2["DIplusN"] + df2["DIminusN"] 
    df2["DIdiff"] =  abs(df2["DIplusN"] - df2["DIminusN"])
    df2["DX"] = 100*(df2["DIdiff"]/df2["DIsum"])
    ADX = []
    DX = df2["DX"].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2["DX"][j-n+1: j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n - 1)*ADX[j-1] + DX[j])/n)
    df2["ADX"] =np.array(ADX)
    return df2["ADX"]

def calTechnicalIndicator(df):
    MSTF_cl = df
    daily_return = MSTF_cl.pct_change()["Close"]
    MSTF_cl["PCT_CHANGE"] = daily_return
    MSTF_cl["MACD"] = MACD(MSTF_cl, 12, 26, 9)["SIGNAL"]
    MSTF_cl["SMA"] = SMA(daily_return, 5)
    MSTF_cl["EMA"] = EMA(daily_return, 5)
    MSTF_cl = ATR(MSTF_cl, 20)
    MSTF_cl = BollBnd(MSTF_cl,10)
    MSTF_cl["RSI"] = np.array(RSI(MSTF_cl, 14))
    MSTF_cl["ADX"] = np.array(ADX(MSTF_cl, 14))
    MSTF_cl = MSTF_cl.dropna(axis=0)
    return MSTF_cl
        
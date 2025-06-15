import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

SAMPLING_RATE = 48000
UPSAMPLE_NUM = 1

# 解析信号の生成
def AnalysisSignal(sgn):
    fft = np.fft.fft(sgn)
    fft[int((len(fft))/2):] = 0
    ifft = np.fft.ifft(fft)
    return ifft

# n次平滑化
def Smoothing(sgn, order):
    smt = []
    for i in range(len(sgn) - (order-1)):
        sum = 0
        for j in range(i, i + (order-1)):
            sum = sum + sgn[j]
        smt.append(sum/order)
    return smt

# チャープ信号の作成
def ChirpMaker(f0, ff, tC, flg):
    t = np.arange(1, tC * SAMPLING_RATE + 1, 1)
    k = (ff - f0) / tC
    phi0 = 0
    chirp = 0.5 * np.sin(phi0 + 2*np.pi*(f0*(t-1)/SAMPLING_RATE + (k/2)*((t-1)/SAMPLING_RATE)**2))

    # ハン窓の設定
    Hann = 0.5 - 0.5 * np.cos(2*np.pi*(t / (tC*SAMPLING_RATE)))

    if (flg == 1):
        chirp = chirp * Hann
        chirp = 0.5 * chirp / np.max(chirp)

    return chirp

# アップサンプリング
def UpSampling(sgn, num):
    upSgn = np.zeros(len(sgn)*num)
    for i in range(len(sgn)):
        upSgn[i*num] = sgn[i]
    fft = np.fft.fft(upSgn)
    fft[int(len(sgn)/2):-int(len(sgn)/2)] = 0
    ifft = np.fft.ifft(fft)
    return np.real(ifft)

# 包絡線の算出
def EnvelopeMaker(chirp, sound, index, wndNum, n, i):
    anlChr = AnalysisSignal(chirp)
    anlSnd = AnalysisSignal(sound[int((index-500)+0.25*i*SAMPLING_RATE):int((index-500)+0.25*(i+1)*SAMPLING_RATE)])

    crl = np.conjugate(np.fft.fft(np.concatenate([anlChr, np.zeros(len(anlSnd)-len(anlChr))]))) * np.fft.fft(anlSnd)
    env = abs(np.fft.ifft(crl))
    env = np.convolve(env[0:int(SAMPLING_RATE/4)], np.ones(11)/11, mode="same")
    env = UpSampling(env,n)[500*n:wndNum*n]

    return env

# スタートタイムの決定
def StartTime(chirp, sound):
    """
    chirp: チャープ信号(np.array)
    sound: 音声信号(np.array)
    直接波の到来タイミングを返す
    """
    anlChr = AnalysisSignal(chirp)
    anlSnd = AnalysisSignal(sound)

    crl = np.conjugate(np.fft.fft(np.concatenate([anlChr, np.zeros(len(anlSnd)-len(anlChr))]))) * np.fft.fft(anlSnd)
    env = abs(np.fft.ifft(crl))
    env = np.convolve(env, np.ones(11)/11, mode="same")
    pak = PeakDetector(env, 16*UPSAMPLE_NUM, 500*UPSAMPLE_NUM, 4)

    return pak[0][0]/SAMPLING_RATE

# CA-CFAR的なピーク検出
def PeakDetector(sgn, delta, grdItv, thrs=1.6):
    # sgn 信号列
    # delta 検出の時間窓長
    # grdItv ガード幅(平均に用いない部分)

    lst = []
    mean = np.mean(sgn)
    strNum = delta + grdItv # インデックスの初期値
    fnsNum = fnsNum = len(sgn) - delta - grdItv # インデックスの最後値

    for i in range(strNum, fnsNum):
        trs = np.mean(sgn[i-grdItv:i+grdItv])*thrs
        if sgn[i]==np.max(sgn[i-delta:i+delta]) and sgn[i]>=mean and sgn[i]>=trs:
            lst.append([i,sgn[i]])
    return lst

def bandpass_filter(x, samplerate, fp, fs, gpass, gstop):
    """
    これは使っていない
    """
    fn = samplerate / 2 #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band") #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x) #信号に対してフィルタをかける
    return y  

def GetGSP(chirp, snd, index_sync, is_train=True, n = 20):
    """
    chirp: チャープ信号(np.array)
    snd: 音声信号(np.array)
    index_sync: チャープ信号と音声信号の大まかな同期時刻(float, s)
    is_train: 学習データか否か(bool)
    n: 取得するチャープ信号の個数(int)
    """

    GSP_train = []
    index = index_sync * SAMPLING_RATE

    # 開始点を学習データなら0s，テストデータなら20sに設定
    if is_train:
        start = 0
    else:
        start = 20

    for i in range(start, start+n):
        env = EnvelopeMaker(chirp, snd, index, 18000, UPSAMPLE_NUM, i)
        pak = PeakDetector(env, 16*UPSAMPLE_NUM, 500*UPSAMPLE_NUM, 4)
        GSP = np.array(env[:int(0.1*SAMPLING_RATE*UPSAMPLE_NUM)])

        GSP_train.append(GSP)
    GSP_train = np.array(GSP_train)
    
    return GSP_train

# 高速フーリエ変換
def calc_amp(data, fs):
    '''フーリエ変換して振幅スペクトルを計算する関数
    '''
    N = len(data)
    window = signal.windows.hann(N)
    F = np.fft.fft(data * window)
    freq = np.fft.fftfreq(N, d=1/fs) # 周波数スケール
    F = F / (N / 2) # フーリエ変換の結果を正規化
    F = F * (N / sum(window)) # 窓関数による振幅減少を補正する
    Amp = np.abs(F) # 振幅スペクトル
    return Amp, freq

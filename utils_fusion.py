import numpy as np
from skimage.transform import pyramid_gaussian
import cv2
from scipy import signal
from cv2.ximgproc import guidedFilter

mat = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])


def la_filter(mono):
    img_shape = mono.shape
    C = np.zeros(img_shape)
    t1 = list([[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]])
    # for i in range(0, img_shape[0]):
    #     for j in range(0, img_shape[1]):
    #         C[i, j] = abs(np.sum(mono[i:i + 3, j:j + 3] * t1))
    myj = signal.convolve2d(mono, t1, mode="same")
    return myj


def contrast(I,exposure_num,img_rows,img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        mono = cv2.cvtColor(I[i].astype(np.float32), cv2.COLOR_BGR2GRAY)
        C[:, :, i] = np.abs(la_filter(mono))

    return C


def saturation(I,exposure_num,img_rows,img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = I[i][:, :, 0]
        G = I[i][:, :, 1]
        B = I[i][:, :, 2]
        mu = (R + G + B) / 3
        C[:, :, i] = np.sqrt(((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)
    return C


def well_exposedness(I,exposure_num,img_rows,img_cols):
    sig = 0.2
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = np.exp(-.4 * (I[i][:, :, 0] - 0.5) ** 2 / sig ** 2)
        G = np.exp(-.4 * (I[i][:, :, 1] - 0.5) ** 2 / sig ** 2)
        B = np.exp(-.4 * (I[i][:, :, 2] - 0.5) ** 2 / sig ** 2)
        C[:, :, i] = R * G * B
    return C


def gaussian_pyramid(I,nlev):
    pyr = []
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2)):
        if i == nlev:
            break
        pyr.append(resized)
    return pyr
    # layer = I.copy()
    # gaussian_pyramid = [layer]
    # for i in range(nlev-1):
    #     layer = cv2.pyrDown(layer)
    #     gaussian_pyramid.append(layer)
    # return gaussian_pyramid


def laplacian_pyramid(I,nlev):
    pyr = []
    J = I
    pyrg = gaussian_pyramid(I,nlev)
    for i in range(0, nlev-1):
        odd = pyrg[i].shape
        temp = pyrg[i] - cv2.resize(pyrg[i+1],(pyrg[i].shape[1], pyrg[i].shape[0]))
        pyr.append(temp)
    pyr.append(pyrg[nlev-1])
    return pyr
    # layer = gaussian_pyramid(I, nlev)
    # laplacian_pyramid = [layer[nlev-1]]
    # for i in range(nlev-1, 0, -1):
    #     size = (layer[i - 1].shape[1], layer[i - 1].shape[0])
    #     gaussian_expanded = cv2.pyrUp(layer[i], dstsize=size)
    #     laplacian = cv2.subtract(layer[i - 1], gaussian_expanded)
    #     laplacian_pyramid.append(laplacian)
    # laplacian_pyramid.reverse()
    # return laplacian_pyramid


def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)
    R = pyr[nlev-1]
    for i in range(nlev-2,-1,-1):
        R = pyr[i] + cv2.resize(R,(pyr[i].shape[1], pyr[i].shape[0]))
    return R


def fusion(uexp, oexp, uYmu = 0.50, uYstd = 0.25, oYmu = 0.50, oYstd = 0.25):
    beta = 2
    vFrTh = 0.16
    RadPr = 3

    I = (uexp, oexp)
    r = uexp.shape[0]
    c = uexp.shape[1]
    n = 2
    nlev = round(np.log(min(r, c)) / np.log(2)) - beta
    nlev = int(nlev)
    RadFr = RadPr * (1<< (nlev - 1))


    W = np.ones((r, c, n))
    I0_gray = cv2.cvtColor(I[0].astype(np.float32), cv2.COLOR_BGR2GRAY)
    I1_gray = cv2.cvtColor(I[1].astype(np.float32), cv2.COLOR_BGR2GRAY)
    Y0 = guidedFilter(guide=I0_gray, src=I0_gray, radius=RadFr, eps=vFrTh)/ 255.0
    Y1 = guidedFilter(guide=I1_gray, src=I1_gray, radius=RadFr, eps=vFrTh)/ 255.0

    W[:,:,0] = np.exp(-((Y0 - uYmu)**2) / (2 * (uYstd**2)))
    W[:, :, 1] = np.exp(-((Y1 - oYmu) ** 2) / (2 * (oYstd ** 2)))

    W = W + 1e-12
    Norm = np.array([np.sum(W, 2), np.sum(W, 2)])
    Norm = Norm.swapaxes(0, 2)
    Norm = Norm.swapaxes(0, 1)
    W = W / Norm

    YC0 = cv2.cvtColor(I[0].astype(np.float32), cv2.COLOR_BGR2YCR_CB)
    YC1 = cv2.cvtColor(I[1].astype(np.float32), cv2.COLOR_BGR2YCR_CB)
    YC = (YC0,YC1)

    II = (uexp / 255.0, oexp/ 255.0)

    pyr = gaussian_pyramid(np.zeros((r, c, 3)), nlev)
    for i in range(0, n):
        pyrw = gaussian_pyramid(W[:, :, i], nlev)
        pyri = laplacian_pyramid(II[i], nlev)
        for ii in range(0, nlev):
            w = np.array([pyrw[ii], pyrw[ii], pyrw[ii]])
            w = w.swapaxes(0, 2)
            w = w.swapaxes(0, 1)
            pyr[ii] = pyr[ii] + w * pyri[ii]
    R = reconstruct_laplacian_pyramid(pyr)
    # R = cv2.cvtColor(R.astype(np.float32), cv2.COLOR_YCR_CB2BGR)
    # R = ycbcr2rgb(R)

    # R = R * 255
    return R


def three1_fusion(uexp, oexp, virtual):
    I = (uexp / 255.0, oexp/ 255.0, virtual/ 255.0)
    r = uexp.shape[0]
    c = uexp.shape[1]
    n = 3
    nlev = round(np.log(min(r, c)) / np.log(2)) - 2
    nlev = int(nlev)

    W = np.ones((r, c, n))

    W = np.multiply(W, contrast(I, n, r, c))
    W = np.multiply(W, saturation(I, n, r, c))
    W = np.multiply(W, well_exposedness(I, n, r, c))

    W = W + 1e-12
    Norm = np.array([np.sum(W, 2), np.sum(W, 2), np.sum(W, 2)])
    Norm = Norm.swapaxes(0, 2)
    Norm = Norm.swapaxes(0, 1)
    W = W / Norm

    YC0 = cv2.cvtColor(I[0].astype(np.float32), cv2.COLOR_RGB2YCR_CB)
    YC1 = cv2.cvtColor(I[1].astype(np.float32), cv2.COLOR_BGR2YCR_CB)
    YC2 = cv2.cvtColor(I[2].astype(np.float32), cv2.COLOR_BGR2YCR_CB)
    YC = (YC0, YC1, YC2)

    pyr = gaussian_pyramid(np.zeros((r, c, 3)), nlev)
    for i in range(0, n):
        pyrw = gaussian_pyramid(W[:, :, i], nlev)
        pyri = laplacian_pyramid(I[i], nlev)
        for ii in range(0, nlev):
            w = np.array([pyrw[ii], pyrw[ii], pyrw[ii]])
            w = w.swapaxes(0, 2)
            w = w.swapaxes(0, 1)
            pyr[ii] = pyr[ii] + w * pyri[ii]
    R = reconstruct_laplacian_pyramid(pyr)
    # R = cv2.cvtColor(R.astype(np.float32), cv2.COLOR_YCR_CB2BGR)


    return R.astype(np.float32)

def gguidedfilter(I, p, r, eps):
    [hei, wid] = I.shape
    rn = 2*r+1
    N = cv2.boxFilter(np.ones(I.shape), -1, (rn,rn))

    mean_I = cv2.boxFilter(I, -1, (rn,rn)) / N
    mean_p = cv2.boxFilter(p, -1, (rn,rn)) / N
    mean_Ip = cv2.boxFilter(I* p, -1, (rn,rn)) / N
    cov_Ip = mean_Ip - mean_I* mean_p
    mean_II = cv2.boxFilter(I* I, -1, (rn,rn)) / N
    var_I = mean_II - mean_I* mean_I
    var_I[var_I < 0] = 0

    r2 = 1
    rn2 = 2*r2+1
    N2 = cv2.boxFilter(np.ones(I.shape), -1, (rn2,rn2))
    mean_I2 = cv2.boxFilter(I, -1, (rn2,rn2)) / N2
    mean_II2 = cv2.boxFilter(I * I, -1, (rn2,rn2)) / N2
    var_I2 = mean_II2 - mean_I2* mean_I2
    var_I2[var_I2 < 0] = 0

    var = (var_I2 * var_I) ** 0.5
    eps0 = (0.001) ** 2
    varfinal = (var + eps0) * np.sum(np.sum(1. / (var + eps0))) / (hei * wid)

    minV = np.min(var)
    meanV = np.mean(var)
    alpha = meanV
    kk = -4 / (minV - alpha)
    w = 1 - 1. / (1 + np.exp(kk * (var - alpha)))

    a = (cov_Ip + eps * w / varfinal) / (var_I + eps / varfinal)
    b = mean_p - a* mean_I

    mean_a = cv2.boxFilter(a, -1, (rn,rn)) / N
    mean_b = cv2.boxFilter(b, -1, (rn,rn)) / N

    q = mean_a * I + mean_b

    return q

def IMF(Ref,A):
    Ref_His = np.zeros((1, 256))
    Img_His = np.zeros((1, 256))
    Ref = Ref.flatten()
    Img = A.flatten()
    for i in range(len(Ref)):
        a = Ref[i]
        Ref_His[0,a] = Ref_His[0,a] + 1
        a = Img[i]
        Img_His[0,a] = Img_His[0,a] + 1
    for i in range(1, 256):
        Ref_His[0,i] = Ref_His[0,i-1] + Ref_His[0,i]
        Img_His[0,i] = Img_His[0,i - 1] + Img_His[0,i]
    IMFTable = np.zeros((1, 256))
    for i in range(256):
        a = np.abs(Ref_His[0,i]-Img_His)
        IMFTable[0,i] = np.argmin(a)

    return IMFTable, Ref_His, Img_His



def IMF_virtual(uexp, oexp):
    n = 2;
    GY = []
    GY.append(uexp[:,:,0])
    Tvalue0 = np.sum(uexp[:,:,0])
    GY.append(oexp[:, :, 0])
    Tvalue1 = np.sum(oexp[:, :, 0])

    I1 = np.zeros(uexp.shape)
    F1 = np.zeros(uexp.shape)

    Ref = GY[1]
    Ref2A = np.zeros((256, n - 1))
    A2Ref = np.zeros((256, 1))
    A = GY[0]
    TableRef2A, HRef, HA = IMF(Ref,A)
    TableA2Ref, HRef1, HA1 = IMF(A, Ref)
    Ref2A = np.transpose(TableRef2A)
    A2Ref = np.transpose(TableA2Ref)

    a = np.array([i for i in range(256)])
    a = np.expand_dims(a, axis=1)
    Ref2A= np.hstack((Ref2A,a))
    A2Ref = np.hstack((A2Ref, a))
    DR2A = np.diff(Ref2A,axis=0)
    DA2R = np.diff(A2Ref,axis=0)

    I = GY[0].copy()
    for i in range(256):
        idx = np.where(GY[0] == A2Ref[i,1]+1)
        if A2Ref[i, 0] == 255:
            I[idx] = A2Ref[i, 0]
        else:
            I[idx] = A2Ref[i, 0]+1
    I1[:,:, 0] = I

    F = GY[1].copy()
    for i in range(256):
        idx = np.where(GY[1] == Ref2A[i,1]+1)
        if Ref2A[i, 0] == 255:
            F[idx] = Ref2A[i, 0]
        else:
            F[idx] = Ref2A[i, 0] + 1
    F1[:,:, 0]=F

    GY = []
    GY.append(uexp[:, :, 1])
    Tvalue0 = np.sum(uexp[:, :, 1])
    GY.append(oexp[:, :, 1])
    Tvalue1 = np.sum(oexp[:, :, 1])

    Ref = GY[1]
    Ref2A = np.zeros((256, n - 1))
    A2Ref = np.zeros((256, 1))
    A = GY[0]
    TableRef2A, HRef, HA = IMF(Ref, A)
    TableA2Ref, HRef1, HA1 = IMF(A, Ref)
    Ref2A = np.transpose(TableRef2A)
    A2Ref = np.transpose(TableA2Ref)

    a = np.array([i for i in range(256)])
    a = np.expand_dims(a, axis=1)
    Ref2A= np.hstack((Ref2A,a))
    A2Ref = np.hstack((A2Ref, a))
    DR2A = np.diff(Ref2A,axis=0)
    DA2R = np.diff(A2Ref,axis=0)

    I = GY[0].copy()
    for i in range(256):
        idx = np.where(GY[0] == A2Ref[i, 1]+1)
        if A2Ref[i, 0] == 255:
            I[idx] = A2Ref[i, 0]
        else:
            I[idx] = A2Ref[i, 0] + 1
    I1[:, :, 1] = I

    F = GY[1].copy()
    for i in range(256):
        idx = np.where(GY[1] == Ref2A[i, 1]+1)
        if Ref2A[i, 0] == 255:
            F[idx] = Ref2A[i, 0]
        else:
            F[idx] = Ref2A[i, 0] + 1
    F1[:, :, 1] = F

    GY = []
    GY.append(uexp[:, :, 2])
    Tvalue0 = np.sum(uexp[:, :, 2])
    GY.append(oexp[:, :, 2])
    Tvalue1 = np.sum(oexp[:, :, 2])

    Ref = GY[1]
    Ref2A = np.zeros((256, n - 1))
    A2Ref = np.zeros((256, 1))
    A = GY[0]
    TableRef2A, HRef, HA = IMF(Ref, A)
    TableA2Ref, HRef1, HA1 = IMF(A, Ref)
    Ref2A = np.transpose(TableRef2A)
    A2Ref = np.transpose(TableA2Ref)

    a = np.array([i for i in range(256)])
    a = np.expand_dims(a, axis=1)
    Ref2A= np.hstack((Ref2A,a))
    A2Ref = np.hstack((A2Ref, a))
    DR2A = np.diff(Ref2A,axis=0)
    DA2R = np.diff(A2Ref,axis=0)

    I = GY[0].copy()
    for i in range(256):
        idx = np.where(GY[0] == A2Ref[i, 1]+1)
        if A2Ref[i, 0] == 255:
            I[idx] = A2Ref[i, 0]
        else:
            I[idx] = A2Ref[i, 0] + 1
    I1[:, :, 2] = I

    F = GY[1].copy()
    for i in range(256):
        idx = np.where(GY[1] == Ref2A[i, 1]+1)
        if Ref2A[i, 0] == 255:
            F[idx] = Ref2A[i, 0]
        else:
            F[idx] = Ref2A[i, 0] + 1
    F1[:, :, 2] = F

    return F1,I1

def fusion11(z):
    w = np.zeros(z.shape)
    for i in range(3):
        temp = z[:,:,i].copy()
        temp = temp.astype(np.float32)
        for ii in range(temp.shape[0]):
            for jj in range(temp.shape[1]):
                if temp[ii,jj] >= 0 and temp[ii,jj]<10:
                    temp[ii, jj] = 0
                elif temp[ii,jj] >= 10 and temp[ii,jj]<55:
                    temp[ii, jj]= 1 - 3. * (((55 - temp[ii, jj])/45)**2)+2.*(((55-temp[ii, jj])/45)**3)
                else:
                    temp[ii, jj] = 1
        w[:,:, i]=temp
    return w

def fusion21(z):
    w = np.zeros(z.shape)
    for i in range(3):
        temp = z[:, :, i].copy()
        temp = temp.astype(np.float32)
        for ii in range(temp.shape[0]):
            for jj in range(temp.shape[1]):
                if temp[ii, jj] >= 0 and temp[ii, jj] < 200:
                    temp[ii, jj] = 1
                elif temp[ii, jj] >= 200 and temp[ii, jj] < 250:
                    temp[ii, jj] = 1-3.*(((temp[ii, jj]-200)/50)**2)+2*(((temp[ii, jj]-200)/50)**3)
                else:
                    temp[ii, jj] = 0
        w[:, :, i] = temp

    return w

def three_fusion(uexp, oexp, virtual):
    vFrTh = 1/1024
    RadPr = 3
    I = (uexp / 255.0, oexp/ 255.0, virtual/ 255.0)
    r = uexp.shape[0]
    c = uexp.shape[1]
    n = 3
    nlev = round(np.log(min(r, c)) / np.log(2)) - 2
    nlev = int(nlev)
    RadFr = 4

    W = np.ones((r, c, n))

    W = np.multiply(W, contrast(I, n, r, c))
    W = np.multiply(W, saturation(I, n, r, c))
    W = np.multiply(W, well_exposedness(I, n, r, c))

    W = W + 1e-12
    Norm = np.array([np.sum(W, 2), np.sum(W, 2), np.sum(W, 2)])
    Norm = Norm.swapaxes(0, 2)
    Norm = Norm.swapaxes(0, 1)
    W = W / Norm

    YC0 = cv2.cvtColor(I[0].astype(np.float32), cv2.COLOR_RGB2YCrCb)
    YC1 = cv2.cvtColor(I[1].astype(np.float32), cv2.COLOR_RGB2YCrCb)
    YC2 = cv2.cvtColor(I[2].astype(np.float32), cv2.COLOR_RGB2YCrCb)
    YC = (YC0[:,:,0], YC1[:,:,0], YC2[:,:,0])

    pyr = gaussian_pyramid(np.zeros((r, c, 3)), nlev)

    for i in range(n):
       temp = YC[i]
       W[:,:,i] =  W[:,:,i]*(np.mean(temp)**2)

    Norm = np.array([np.sum(W, 2), np.sum(W, 2), np.sum(W, 2)])
    Norm = Norm.swapaxes(0, 2)
    Norm = Norm.swapaxes(0, 1)
    W = W / Norm

    wsum = gaussian_pyramid(np.zeros((r, c)), nlev)

    for i in range(n):
        pyrw = gaussian_pyramid(W[:, :, i], nlev)
        pyry = gaussian_pyramid(YC[i], nlev)
        pyri = laplacian_pyramid(I[i], nlev)


        for ii in range(nlev):
            pyrw[ii] = gguidedfilter(pyry[ii], pyrw[ii], RadFr, vFrTh)

        for ii in range(nlev):
            if i==0:
                wsum[ii] = pyrw[ii] + 1e-12
            else:
                wsum[ii] = wsum[ii] + pyrw[ii] + 1e-12
            w = pyrw[ii] + 1e-12
            wNorm = np.array([w, w, w])
            wNorm = wNorm.swapaxes(0, 2)
            wNorm = wNorm.swapaxes(0, 1)

            pyr[ii] = wNorm*pyri[ii] + pyr[ii]


    for ii in range(nlev):
        ws = np.array([wsum[ii], wsum[ii], wsum[ii]])
        ws = ws.swapaxes(0, 2)
        ws = ws.swapaxes(0, 1)

        pyr[ii] = pyr[ii] / ws


    # for i in range(0, n):
    #     pyrw = gaussian_pyramid(np.expand_dims(W[:, :, i], axis=2), nlev)
    #     pyri = laplacian_pyramid(I[i], nlev)
    #     for ii in range(0, nlev):
    #         w = np.squeeze(np.array([pyrw[ii], pyrw[ii], pyrw[ii]]))
    #         w = w.swapaxes(0, 2)
    #         w = w.swapaxes(0, 1)
    #         pyr[ii] = pyr[ii] + np.multiply(w, pyri[ii])
    R = reconstruct_laplacian_pyramid(pyr)
    # R = cv2.cvtColor(R.astype(np.float32), cv2.COLOR_YCR_CB2BGR)

    return R.astype(np.float32)

def fusion_virtual(I2,I1):
    h_s, s_h = IMF_virtual(I1, I2)
    w1 = fusion11(I1)
    w2 = fusion21(I2)
    Is3 = (I1 * s_h)** (0.5)
    Ih3 = (I2 * h_s)**(0.5)
    final_virtual = (w1 * Is3 + w2 * Ih3) / (w1 + w2+0.00000000000000001)
    final_virtual = np.uint8(final_virtual)
    final_fusion = three_fusion(I1, I2, final_virtual)
    return final_fusion


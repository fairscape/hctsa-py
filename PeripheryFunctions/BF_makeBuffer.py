def BF_makeBuffer(y, bufferSize):

    N = len(y)

    numBuffers = int(np.floor(N / bufferSize))

    y_buffer = y[0:numBuffers*bufferSize]

    y_buffer = y_buffer.reshape((numBuffers,bufferSize))

    return y_buffer

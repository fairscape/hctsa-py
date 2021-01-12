#© 2020 By The Rector And Visitors Of The University Of Virginia

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def CO_tc3(y,tau = 'ac'):

    if tau == 'ac':

        tau = CO_FirstZero(y,'ac')

    elif tau == 'mi':

        tau = CO_FirstMin(y,'mi')

    N = len(y)

    yn = y[0:N-2*tau]
    yn1 = y[tau:N-tau]
    yn2 = y[tau*2:N]

    try:

        raw = np.mean(np.multiply(np.multiply(yn,yn1),yn2)) / (np.absolute(np.mean(np.multiply(yn,yn1))) ** (3/2))

    except:

        return({'raw':np.nan,'abs':np.nan,'num':np.nan,
                'absnum':np.nan,'denom':np.nan})

    outDict = {}

    outDict['raw'] = raw

    outDict['abs'] = np.absolute(raw)

    outDict['num'] = np.mean(np.multiply(yn,np.multiply(yn1,yn2)))

    outDict['absnum'] = np.absolute(outDict['num'])

    outDict['denom'] = np.absolute( np.mean(np.multiply(yn,yn1)))**(3/2)

    return outDict

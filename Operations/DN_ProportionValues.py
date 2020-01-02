def DN_ProportionValues(x,propWhat = 'positive'):
    N = len(x)
    if propWhat == 'zeros':
        return sum(x == 0) / N
    elif propWhat == 'positive':
        return sum(x > 0) / N
    elif propWhat == 'negative':
        return sum(x < 0) / N
    else:
        raise Exception('Only negative, positve, zeros accepted for propWhat.')

def analyze_while_loop(cond_fun, body_fun, init_val):
    flag = cond_fun(init_val)
    while flag:
        init_val = body_fun(init_val)
        flag = cond_fun(init_val)
    return init_val


def analyze_fori_loop(lower, upper, body_fun, init_val):
    for i in range(lower, upper):
        init_val = body_fun(i, init_val)
    return init_val

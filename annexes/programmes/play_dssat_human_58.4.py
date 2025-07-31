def play(state):
    if state.days_planting <= 39:
        return {"fertilizer_quant": 0}
    else:
        if state.days_planting <= 40:
            return {"fertilizer_quant": 27}
        else:
            if state.days_planting <= 44:
                return {"fertilizer_quant": 0}
            else:
                if state.days_planting <= 45:
                    return {"fertilizer_quant": 35}
                else:
                    if state.days_planting <= 79:
                        return {"fertilizer_quant": 0}
                    else:
                        if state.days_planting <= 80:
                            return {"fertilizer_quant": 54}
                        else:
                            return {"fertilizer_quant": 0}
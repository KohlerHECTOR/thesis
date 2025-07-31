def play(state):
    if state.nitrogen - state.days_planting  <= -17.50:
        if state.nitrogen - state.water_stress  <= 13.50:
            if state.nitrogen - state.days_planting  <= -39.50:
                if state.days_planting - state.growing_degree   <= -5.00:
                    return {"fertilizer_quant": 0.0, }
                else:
                    return {"fertilizer_quant": 27.0, }
            else:
                return {"fertilizer_quant": 0.0, }
        else:
            if state.plant_transpi - state.maize_growing  <= -930.64:
                return {"fertilizer_quant": 54.0, }
            else:
                return {"fertilizer_quant": 35.0, }
    else:
        return {"fertilizer_quant": 0.0, }

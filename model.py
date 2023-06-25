from pydantic import BaseModel

class iyzico(BaseModel):
    sales_roll_mean_91: float 
    sales_roll_mean_92: float
    sales_roll_mean_360:float
    sales_roll_mean_182:float 
    day_of_week_6: int 
    sales_lag_91: float 
    sales_lag_364:float 
    sales_ewm_alpha_095_lag_91: float
    sales_ewm_alpha_09_lag_91: float
    sales_roll_mean_178:float
    sales_roll_mean_179:float
    sales_roll_mean_181:float 
   
    
    
    
    class Config:
        schema_extra = {
            "example": {
                "sales_roll_mean_91": 1752.490,  
                "sales_roll_mean_92": 1751.024,
                "sales_roll_mean_360": 1712.203,
                "sales_roll_mean_182": 1726.533,
                "day_of_week_6": 0,
                "sales_lag_91": 2493.808,
                "sales_lag_364": 1982.756,
                "sales_ewm_alpha_095_lag_91": 2531.099,
                "sales_ewm_alpha_09_lag_91": 2557.298,
                "sales_roll_mean_178": 1723.638,
                "sales_roll_mean_179": 1726.740,
                "sales_roll_mean_181": 1728.182

            }

        }

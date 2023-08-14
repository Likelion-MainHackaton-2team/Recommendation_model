import polars as pl
import numpy as np

class BudgetAnaylsis:
    def __init__(self):
        pass

    def predict(self, data):
        money_total = self.__spend_money_total(data)
        money_by_category = self.__spend_money_by_category(data)
        money_by_month = self.__spend_money_by_month(data)
        money_by_month_category = self.__spend_money_by_month_category(data)

        return {
            'money_total': money_total,
            'money_by_category': money_by_category,
            'money_by_month': money_by_month,
            'money_by_month_category': money_by_month_category
        }

    def __spend_money_total(self, data):
        return data['amount'].sum()

    def __spend_money_by_category(self, data):
        return data.groupby('category').sum()[['Category', 'amount']]

    def __spend_money_by_month(self, data):
        return data.groupby('month').sum()[['month', 'amount']]

    def __spend_money_by_month_category(self, data):
        return data.groupby(['month', 'category']).sum()[['month', 'category', 'amount']]
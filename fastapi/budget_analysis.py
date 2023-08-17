import polars as pl
import numpy as np

class BudgetAnaylsis:
    def predict(self, data):
        amount = data['amount'].to_list()
        amount = pl.Series([int(i) for i in amount])
        data = data.replace("amount", amount)

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
        convert_to_list = data['amount'].to_list()
        convert_to_list = [int(i) for i in convert_to_list]
        return sum(convert_to_list)

    def __spend_money_by_category(self, data):
        resp = data.groupby('category').agg(pl.col('amount').sum())
        resp = resp.to_numpy().tolist()
        return resp

    def __spend_money_by_month(self, data):
        resp = data.groupby('month').sum()[['month', 'amount']]
        resp = resp.to_numpy().tolist()
        return resp

    def __spend_money_by_month_category(self, data):
        resp = data.groupby(['month', 'category']).sum()[['month', 'category', 'amount']]
        resp = resp.to_numpy().tolist()
        return resp
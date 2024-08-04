import pandas as pd  # pandasをインポート
# 遷移確率のCSVファイルを読み込む
change_probability = pd.read_csv('senni_pro.csv')
variable = pd.read_csv('Variable.csv')

print(change_probability)
print(variable)


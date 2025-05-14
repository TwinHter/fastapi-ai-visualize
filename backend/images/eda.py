import io
import matplotlib.pyplot as plt
import pandas as pd
def eda_area_item(area: str, item: str):
    df = pd.read_csv('./models/yield_df.csv')
    plt.figure(figsize=(10, 6))
    temp_df = df[(df['Area'] == area) & (df['Item'] == item)]
    plt.plot(temp_df['Year'], temp_df['hg/ha_yield'])
    plt.title(f'hg/ha_yield over the years for {item} in {area}')
    plt.xlabel('Year')
    plt.ylabel('hg/ha_yield')
    plt.grid(True)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    return img_buf

def eda_area(area: str):
    df_pes = pd.read_csv('./models/pesticides.csv')
    plt.figure(figsize=(10, 6))
    temp_df = df_pes[df_pes['Area'] == area]
    plt.plot(temp_df['Year'], temp_df['Value'])
    plt.title(f'Pesticides Tonnes over the years in {area}')
    plt.xlabel('Year')
    plt.ylabel('Pesticides Tonnes')
    plt.grid(True)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    return img_buf
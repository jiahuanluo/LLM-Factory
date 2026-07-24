"""Home Credit Default Risk 数据集下载与解压

数据源：hf-mirror.com 镜像的 algcache/HomeCreditDefaultRisk（Kaggle 原版 zip）
官方页：https://www.kaggle.com/c/home-credit-default-risk

输出到 data/home-credit/：
  - application_train.csv             主表·训练 (~166 MB, 307k 行)
  - application_test.csv              主表·测试  (~26 MB, 48k 行)
  - bureau.csv                        外部信贷局记录
  - bureau_balance.csv                信贷局月度余额
  - POS_CASH_balance.csv              POS/现贷月度余额
  - credit_card_balance.csv           信用卡月度余额
  - installments_payments.csv         分期付款历史
  - previous_application.csv          Home Credit 历史申请
  - sample_submission.csv
  - HomeCredit_columns_description.csv  字段说明
"""
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

URL = 'https://hf-mirror.com/datasets/algcache/HomeCreditDefaultRisk/resolve/main/home-credit-default-risk.zip'
OUT = Path('data/home-credit')
ZIP_PATH = OUT / 'home-credit-default-risk.zip'
MIN_ZIP_SIZE = 700_000_000  # 期望 ~688 MB；低于 700B 视为异常


def download():
    OUT.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > MIN_ZIP_SIZE:
        print(f'已存在 {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MB)，跳过下载')
        return

    print(f'下载 {URL}')
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        print(f'文件大小: {total / 1024 / 1024:.1f} MB')
        with open(ZIP_PATH, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, unit_divisor=1024,
            desc='下载', ncols=80,
        ) as bar:
            while True:
                buf = resp.read(1024 * 1024)
                if not buf:
                    break
                f.write(buf)
                bar.update(len(buf))

    if ZIP_PATH.stat().st_size < MIN_ZIP_SIZE:
        raise RuntimeError(
            f'下载文件过小：{ZIP_PATH.stat().st_size} bytes (< {MIN_ZIP_SIZE})'
        )


def extract():
    print(f'解压到 {OUT}/')
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(OUT)
    print('解压完成')


def cleanup_zip():
    ZIP_PATH.unlink()
    print(f'已删除 zip: {ZIP_PATH}')


def list_outputs():
    print('\n输出文件：')
    for p in sorted(OUT.glob('*.csv')):
        print(f'  {p.name:40s} {p.stat().st_size / 1024 / 1024:>8.1f} MB')


if __name__ == '__main__':
    download()
    extract()
    list_outputs()
    cleanup_zip()

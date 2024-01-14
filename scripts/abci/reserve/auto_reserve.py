import schedule
import time
import subprocess


def execute_command() -> None:
    try:
        subprocess.run(["qrsub", "-a", "20240122", "-d", "4", "-g", "gcd50698", "-n", "24", "-N", "mambda", "-l", "rt_AF"], check=True)
        print("コマンドが正常に実行されました")
    except subprocess.CalledProcessError as e:
        print(f"コマンドの実行中にエラーが発生しました: {e}")

# 毎日午前10時にコマンドを実行するようにスケジュール設定
schedule.every().day.at("10:00").do(execute_command)

while True:
    schedule.run_pending()
    time.sleep(0.1)

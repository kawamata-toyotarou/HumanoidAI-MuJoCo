# HumanoidAI

Windows上でMuJoCoの人型ロボットを強化学習させるための実験アプリです。Gymnasium `Humanoid-v5`、Stable-Baselines3 PPO、VecNormalizeを使い、二足歩行に向けた報酬設計とカリキュラム学習を検証します。

現在は安定歩行には未達です。フェーズ1からフェーズ3.1までの実験では、前進・姿勢維持の一部指標は改善しましたが、転倒率はまだ高く、歩行完成とは判断していません。

## 現在の範囲

- Python、MuJoCo、Gymnasium、CPU、RAM、GPUの確認
- Python仮想環境 `.venv_phase1` の作成
- Gymnasium `Humanoid-v5` の生成確認
- ランダムな関節制御によるシミュレーション確認
- 3Dウィンドウ表示用の実行スクリプト
- Stable-Baselines3 PPOによる短時間学習
- 学習済みモデルの保存、再開、再生
- TensorBoardログ出力
- VecNormalizeあり/なしのPPO対照実験
- 報酬成分、生存ステップ、前方移動距離、転倒率の評価
- カリキュラム学習Stage 1の実装と転倒原因分析

フェーズ3のGUIはまだ実装していません。

## 作業環境の実測結果

2026-10-12にCodexデスクトップアプリから確認しました。

| 項目 | 結果 |
| --- | --- |
| 指定Python | Windows上のPython 3.14 |
| Pythonバージョン | 3.14.6 |
| 仮想環境 | `.venv_phase1` |
| Gymnasium | 1.4.0 |
| MuJoCo | 3.15.0 |
| Stable-Baselines3 | 2.9.0 |
| PyTorch | 2.14.1+cpu |
| TensorBoard | 2.21.0 |
| Humanoid-v5 | resetと1 stepに成功 |
| 観測 shape | `(348,)` |
| action shape | `(17,)` |
| CPU | 13th Gen Intel(R) Core(TM) i5-1334U、論理CPU 12 |
| RAM | 約15.73 GB、確認時の空き約3.09 GB |
| GPU | Intel(R) Iris(R) Xe Graphics |
| CUDA | PyTorchで未検出。`torch.cuda.is_available()` は `False` |
| NVIDIA GPU | `nvidia-smi` が見つからないため未検出 |

補足: もともとのグローバルPythonにはMuJoCo 3.14.0が入っていました。`gymnasium[mujoco]` の追加時に仮想環境側へMuJoCo 3.15.0も入り、フェーズ1の実行では仮想環境側の3.15.0を使っています。

## セットアップ手順

PowerShellでリポジトリのフォルダに移動します。

```powershell
cd HumanoidAI-MuJoCo
```

今回作成した仮想環境を使います。

```powershell
.\.venv_phase1\Scripts\Activate.ps1
python -m humanoid_ai.check_system
```

仮想環境を作り直す場合は、既存ファイルを削除せず別名で作ることを推奨します。今回実行した作成手順は次の通りです。

```powershell
py -3.14 -m venv --system-site-packages --without-pip .venv_phase1
py -3.14 -m pip install --target .\.venv_phase1\Lib\site-packages "gymnasium[mujoco]"
py -3.14 -m pip install --target .\.venv_phase1\Lib\site-packages stable-baselines3 tensorboard
```

通常の`venv`作成では`ensurepip`が失敗したため、`.venv_phase1`は`--without-pip`で作成し、グローバル側のpipから仮想環境の`site-packages`へ依存関係を配置しています。

## ランダム制御の実行

3Dウィンドウで確認する場合:

```powershell
.\.venv_phase1\Scripts\python.exe -m humanoid_ai.phase1_random_humanoid --steps 1000 --render-mode human
```

画面表示なしで動作確認する場合:

```powershell
.\.venv_phase1\Scripts\python.exe -m humanoid_ai.phase1_random_humanoid --steps 200 --render-mode none
```

オフスクリーン画像レンダリングを確認する場合:

```powershell
.\.venv_phase1\Scripts\python.exe -m humanoid_ai.phase1_random_humanoid --steps 20 --render-mode rgb_array
```

## 今回の確認結果

ヘッドレス実行では、`Humanoid-v5` の生成、`reset(seed=123)`、ランダムactionによる`step()`が成功しました。

`--render-mode rgb_array` では20 step実行し、`(480, 480, 3)` のRGBフレーム取得に成功しました。

`--render-mode human` では120 step実行し、MuJoCo/GLFW由来のエラーなしで完走しました。ただし、Codexから起動した3Dウィンドウが実際にユーザー画面へ表示されたかは目視確認できていません。見えなかった場合は、このREADMEの3Dウィンドウ確認コマンドをPowerShellから直接実行してください。

## フェーズ2: PPO学習

学習中は3D描画を無効にし、CPUでも動く設定にしています。GPU/CUDAが使える場合は`--device auto`または`--device cuda`を指定できますが、今回の確認環境ではCUDAは未検出だったためCPUで検証しました。

10,000ステップの短い学習テスト:

```powershell
.\.venv_phase1\Scripts\python.exe train.py --total-timesteps 10000 --run-name phase2_10000_test --checkpoint-dir checkpoints --log-dir runs\tensorboard --device cpu
```

PPOは内部のロールアウト単位で進むため、10,000指定時の実測保存ステップは10,240でした。これは正常です。

主なCLI引数:

- `--total-timesteps`: 今回実行する学習ステップ数
- `--resume`: 既存の`.zip`モデルから再開
- `--checkpoint-dir`: 保存先。既定は`checkpoints`
- `--log-dir`: TensorBoardログ保存先。既定は`runs/tensorboard`
- `--device`: `auto`、`cpu`、`cuda`
- `--save-freq`: 途中チェックポイントの保存間隔

学習を再開する例:

```powershell
.\.venv_phase1\Scripts\python.exe train.py --resume checkpoints\humanoid_ppo_latest.zip --total-timesteps 10000 --run-name phase2_resume --checkpoint-dir checkpoints --log-dir runs\tensorboard --device cpu
```

保存モデルの再生:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\humanoid_ppo_latest.zip --render-mode human --deterministic --device cpu
```

ランダム動作との比較:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent random --render-mode human
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\humanoid_ppo_latest.zip --render-mode human --deterministic --device cpu
```

画面なしで報酬とエピソード長だけ比較する場合:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent random --render-mode none --episodes 2 --max-steps 300 --seed 123
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\humanoid_ppo_latest.zip --render-mode none --episodes 2 --max-steps 300 --seed 123 --deterministic --device cpu
```

今回の画面なし比較では、ランダム動作は報酬約104/150、長さ22/31 step、学習済みモデルは報酬約268/260、長さ50/49 stepでした。10,000ステップは動作確認用であり、歩行の習得を保証するものではありません。

TensorBoard:

```powershell
.\.venv_phase1\Scripts\python.exe -m tensorboard.main --logdir runs\tensorboard
```

ブラウザで表示されたURLを開くと、`rollout/ep_rew_mean`、`rollout/ep_len_mean`、`train/loss`、`train/approx_kl`などを確認できます。

## フェーズ2の確認結果

- Stable-Baselines3 2.9.0、PyTorch 2.14.1+cpu、TensorBoard 2.21.0のimportに成功しました。
- `train.py`の512ステップ smoke test に成功しました。
- `train.py`の10,000ステップ指定テストに成功し、実測10,240ステップで保存されました。
- `checkpoints\humanoid_ppo_latest.zip` と `checkpoints\humanoid_ppo_steps_10240.zip` を保存しました。
- `runs\tensorboard\phase2_10000_test_1` にTensorBoardイベントファイルが生成されました。
- `play.py --agent model --render-mode none` で保存モデルのロードと推論に成功しました。
- `play.py --agent model --render-mode rgb_array` で`(480, 480, 3)`のRGBフレーム取得に成功しました。
- `play.py --agent model --render-mode human` はエラーなしで完走しました。
- `--resume checkpoints\humanoid_ppo_latest.zip` による512ステップの再開学習に成功し、10,752ステップのモデルを保存できました。

未確認:

- Codexから起動した`--render-mode human`の3Dウィンドウが、ユーザーの画面に実際に表示されたかは目視確認できていません。
- 長時間の本格学習で安定した二足歩行を習得するかは未確認です。本格学習は実行前に確認してください。

## 評価とCPU速度計測

既存の学習済みモデルを上書きしないよう、長めの作業前には`checkpoints`をバックアップしてください。今回のバックアップ先:

```text
backups\before_benchmark_20261012_0100
```

評価スクリプト:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\humanoid_ppo_latest.zip --episodes 10 --max-steps 1000 --seed 2000 --deterministic --device cpu --csv work\eval_current_model.csv
.\.venv_phase1\Scripts\python.exe evaluate.py --agent random --episodes 10 --max-steps 1000 --seed 2000 --csv work\eval_random.csv
```

評価指標:

- `reward_mean`: 平均報酬
- `survival_steps_mean`: 平均生存ステップ数
- `forward_distance_mean`: 胴体のx座標から計算した平均前方移動距離
- `fall_rate`: `terminated=True`で終了した割合
- `truncated_rate`: 最大ステップ到達で打ち切られた割合

今回の評価結果:

| 対象 | 平均報酬 | 平均生存step | 平均前方移動距離 | 転倒率 |
| --- | ---: | ---: | ---: | ---: |
| ランダム | 約111.18 | 23.6 | 約-0.04 | 100% |
| 現在モデル | 約290.69 | 54.4 | 約0.63 | 100% |

CPU速度計測は、学習済みモデルを`checkpoints`から読み込み、保存先を`work\benchmark_checkpoints`に分けて実行しました。3D描画は無効です。

```powershell
.\.venv_phase1\Scripts\python.exe train.py --resume checkpoints\humanoid_ppo_latest.zip --total-timesteps 5000 --run-name benchmark_5000_cpu --checkpoint-dir work\benchmark_checkpoints --log-dir work\benchmark_tensorboard --save-freq 2500 --device cpu
```

実測結果:

- 指定ステップ: 5,000
- 実行ステップ: 5,120
- 経過時間: 7.830秒
- 速度: 約653.854 steps/sec
- PyTorch CPU threads: 10
- CUDA: 利用不可

この速度からの単純推定:

| 学習ステップ | 推定時間 |
| ---: | ---: |
| 100,000 | 約2分33秒 |
| 500,000 | 約12分45秒 |
| 1,000,000 | 約25分29秒 |

注意: 長時間実行ではCPU温度、電源設定、バックグラウンド処理により速度が落ちる可能性があります。発熱を抑えたい場合は`--torch-threads 6`のようにスレッド数を制限できますが、学習速度は下がる可能性があります。

## 実験: 100,000ステップ追加学習

2026-10-12に、現在モデル`checkpoints\humanoid_ppo_latest.zip`から追加で100,000ステップ学習しました。既存の`checkpoints`直下のモデルとバックアップは上書きせず、結果は`checkpoints\experiment_100k`へ保存しています。学習中の3D描画は無効です。

学習コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe train.py --resume checkpoints\humanoid_ppo_latest.zip --total-timesteps 100000 --run-name experiment_100k_from_10240_cpu --checkpoint-dir checkpoints\experiment_100k --final-name-prefix experiment_100k_from_10240_cpu --log-dir runs\tensorboard --save-freq 25000 --device cpu
```

実験条件:

| 項目 | 値 |
| --- | --- |
| 開始モデル | `checkpoints\humanoid_ppo_latest.zip` |
| 追加学習先 | `checkpoints\experiment_100k` |
| 実行デバイス | CPU |
| PyTorch threads | 10 |
| 3D描画 | 無効 |
| PPO設定 | `n_steps=512`, `batch_size=64`, `n_epochs=5`, `learning_rate=3e-4` |
| TensorBoard | `runs\tensorboard\experiment_100k_from_10240_cpu_0` |

学習結果:

| 項目 | 値 |
| --- | ---: |
| 指定追加ステップ | 100,000 |
| 実行追加ステップ | 100,352 |
| 総ステップ | 110,592 |
| 実学習時間 | 153.820秒 |
| 実測速度 | 652.399 steps/sec |

保存モデル:

- `checkpoints\experiment_100k\experiment_100k_from_10240_cpu_latest.zip`
- `checkpoints\experiment_100k\experiment_100k_from_10240_cpu_steps_110592.zip`

学習前後の評価は、学習環境とは別プロセスの評価環境で実行しました。評価条件は同一で、`seed=3000`から20エピソード、最大1000 step、決定論的推論です。

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\humanoid_ppo_latest.zip --episodes 20 --max-steps 1000 --seed 3000 --deterministic --device cpu --csv work\experiment_100k_eval_before.csv
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\experiment_100k\experiment_100k_from_10240_cpu_latest.zip --episodes 20 --max-steps 1000 --seed 3000 --deterministic --device cpu --csv work\experiment_100k_eval_after.csv
```

評価結果:

| 対象 | 平均報酬 | 平均生存step | 平均前方移動距離 | 転倒率 |
| --- | ---: | ---: | ---: | ---: |
| 学習前 | 283.70 | 53.05 | 0.618 | 100% |
| 100k追加後 | 333.00 | 64.35 | 0.616 | 100% |

平均報酬と平均生存ステップは改善しましたが、20エピソードすべて最終的には転倒しています。平均前方移動距離はほぼ横ばいです。安定歩行の習得はまだ未達です。

再生確認:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\experiment_100k\experiment_100k_from_10240_cpu_latest.zip --render-mode none --episodes 1 --max-steps 200 --seed 3000 --deterministic --device cpu
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\experiment_100k\experiment_100k_from_10240_cpu_latest.zip --render-mode rgb_array --episodes 1 --max-steps 20 --seed 3000 --deterministic --device cpu
```

3Dウィンドウで確認する場合:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\experiment_100k\experiment_100k_from_10240_cpu_latest.zip --render-mode human --episodes 1 --max-steps 1000 --seed 3000 --deterministic --device cpu --sleep 0.005
```

Codexからは3Dウィンドウの目視確認はできないため、歩行の見た目は上記コマンドをPowerShellから実行して確認してください。今回は100,000ステップ追加学習までで停止し、それ以降の長時間学習は開始していません。

## フェーズ2.5: 前進学習の調査とVecNormalize実験

現在のPPO設定:

| 項目 | 値 |
| --- | --- |
| アルゴリズム | Stable-Baselines3 PPO |
| policy | `MlpPolicy` |
| `n_steps` | 512 |
| `batch_size` | 64 |
| `n_epochs` | 5 |
| `learning_rate` | 3e-4 |
| device | CPU |
| PyTorch threads | 10 |
| 学習時描画 | 無効 |

Humanoid-v5の報酬構造:

```text
reward = reward_survive + reward_forward + reward_ctrl + reward_contact
reward_forward = forward_reward_weight * x_velocity
reward_ctrl = -control_cost
reward_contact = -contact_cost
```

実環境の`info`には、`reward_survive`、`reward_forward`、`reward_ctrl`、`reward_contact`、`x_velocity`、`y_velocity`、`x_position`などが含まれていることを確認しました。`evaluate.py`はこれらの成分をエピソード単位で集計し、CSVにも保存します。

報酬成分つき評価:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\experiment_100k\experiment_100k_from_10240_cpu_latest.zip --episodes 20 --max-steps 1000 --seed 3000 --deterministic --device cpu --csv work\phase25_component_eval.csv
```

VecNormalize対応:

- `train_phase25.py` を追加しました。
- `--vec-normalize` を付けると観測と報酬をVecNormalizeで正規化します。
- 正規化統計は`*_vecnormalize.pkl`としてモデルと同じフォルダへ保存します。
- `evaluate.py`と`play.py`は`--vec-normalize-stats`で統計を読み込めます。
- 既存モデルへVecNormalizeを後付けすることはしません。正規化あり実験は新規モデルから開始します。

短い動作確認:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 1024 --seed 4242 --run-name phase25_smoke_raw --checkpoint-dir work\phase25_smoke_raw --final-name-prefix phase25_smoke_raw --log-dir work\phase25_tensorboard --save-freq 512 --device cpu
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 1024 --seed 4242 --run-name phase25_smoke_vecnorm --checkpoint-dir work\phase25_smoke_vecnorm --final-name-prefix phase25_smoke_vecnorm --log-dir work\phase25_tensorboard --save-freq 512 --device cpu --vec-normalize
```

確認結果:

- 正規化なし1024ステップ学習に成功しました。
- VecNormalizeあり1024ステップ学習に成功しました。
- VecNormalize統計`work\phase25_smoke_vecnorm\phase25_smoke_vecnorm_vecnormalize.pkl`の保存に成功しました。
- `evaluate.py --vec-normalize-stats ...`で正規化統計を読み込んだ評価に成功しました。
- `play.py --vec-normalize-stats ... --render-mode rgb_array`で`(480, 480, 3)`のフレーム取得に成功しました。
- TensorBoardログは`work\phase25_tensorboard`に生成されました。

100,000ステップ x 2本の対照実験案:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase25_raw_100k_seed4242 --checkpoint-dir checkpoints\phase25_raw_100k --final-name-prefix phase25_raw_100k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase25_vecnorm_100k_seed4242 --checkpoint-dir checkpoints\phase25_vecnorm_100k --final-name-prefix phase25_vecnorm_100k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize
```

評価コマンド案。同じseed列、20エピソード、最大1000 stepで比較します。

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase25_raw_100k\phase25_raw_100k_seed4242_latest.zip --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --csv work\phase25_raw_100k_eval.csv
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_vecnormalize.pkl --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --csv work\phase25_vecnorm_100k_eval.csv
```

推定時間:

- 正規化なし100,000ステップ: 約2分35秒から3分
- VecNormalizeあり100,000ステップ: 約3分から4分
- 合計: 約6分から7分程度

上記は短いsmoke testと直近100k実験の実測からの推定です。CPU温度やバックグラウンド処理で変動します。

### フェーズ2.5 対照実験の実行結果

2026-10-12に、承認後に以下の2本を実行しました。どちらも新規モデルから開始し、既存の`checkpoints`内のモデルは上書きしていません。学習時の3D描画は無効、CPU実行、seedは`4242`、PPOハイパーパラメータは同一です。

| 実験 | VecNormalize | 保存先 | 実ステップ | 学習時間 | steps/sec |
| --- | --- | --- | ---: | ---: | ---: |
| A | なし | `checkpoints\phase25_raw_100k` | 100,352 | 158.926秒 | 631.438 |
| B | あり | `checkpoints\phase25_vecnorm_100k` | 100,352 | 179.483秒 | 559.118 |

保存ファイル:

```text
checkpoints\phase25_raw_100k\phase25_raw_100k_seed4242_latest.zip
checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_latest.zip
checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_vecnormalize.pkl
```

TensorBoardログ:

```text
runs\tensorboard\phase25_raw_100k_seed4242_1
runs\tensorboard\phase25_vecnorm_100k_seed4242_1
```

TensorBoard起動:

```powershell
.\.venv_phase1\Scripts\tensorboard.exe --logdir runs\tensorboard
```

評価条件:

| 項目 | 値 |
| --- | --- |
| 評価エピソード数 | 20 |
| 評価seed列 | `6000`から`6019` |
| 最大step | 1000 |
| 方策 | deterministic |
| 評価環境 | 学習環境とは別に作成 |
| VecNormalize評価 | 保存済み統計を読み込み、統計更新なし、報酬正規化なし |

評価結果:

| 実験 | 平均報酬 | 平均生存step | 平均前方移動距離 | 転倒率 |
| --- | ---: | ---: | ---: | ---: |
| A: VecNormalizeなし | 328.62 | 65.95 | 0.423 | 100% |
| B: VecNormalizeあり | 370.75 | 68.80 | 0.794 | 100% |

報酬成分の平均:

| 実験 | survive | forward | ctrl | contact | x_velocity | y_velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: VecNormalizeなし | 324.75 | 24.57 | -9.99 | -10.71 | 19.65 | -21.42 |
| B: VecNormalizeあり | 339.00 | 55.28 | -12.39 | -11.13 | 44.22 | -7.71 |

評価CSV:

```text
work\phase25_raw_100k_eval.csv
work\phase25_vecnorm_100k_eval.csv
```

今回の100,000ステップ対照実験では、VecNormalizeありのほうが平均報酬、生存step、前方移動距離、forward報酬のすべてで良い結果でした。特に平均前方移動距離は`0.423`から`0.794`へ増え、前進方向の学習にはVecNormalizeが有効そうです。ただし、どちらも20エピソードすべてで転倒しており、安定した歩行の習得にはまだ到達していません。

3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase25_raw_100k\phase25_raw_100k_seed4242_latest.zip --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_vecnormalize.pkl --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

Codex上では3Dウィンドウの目視確認は行っていません。代わりに、両モデルとも`--render-mode rgb_array --max-steps 5`で`(480, 480, 3)`の描画フレーム取得まで確認しました。追加の長時間学習は開始していません。

### フェーズ2.5 VecNormalizeモデルの継続学習100,000ステップ

2026-10-12に、VecNormalizeあり100,000ステップモデルから追加100,000ステップの継続学習を実行しました。元モデルと元の正規化統計は上書きせず、新しいフォルダへ保存しました。

継続元:

```text
checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_latest.zip
checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_vecnormalize.pkl
```

実行コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase25_vecnorm_continue_100k_seed4242 --checkpoint-dir checkpoints\phase25_vecnorm_continue_100k --final-name-prefix phase25_vecnorm_continue_100k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --resume checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase25_vecnorm_100k\phase25_vecnorm_100k_seed4242_vecnormalize.pkl
```

再開確認:

| 項目 | 結果 |
| --- | --- |
| 開始時累計step | 100,352 |
| PPO optimizer状態 | `optimizer_state_entries=13`で非空 |
| VecNormalize統計更新 | `training=True` |
| 報酬正規化 | `norm_reward=True` |
| 学習時描画 | 無効 |
| CPU実行 | `device=cpu` |

保存先:

```text
checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_latest.zip
checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_steps_200704.zip
checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_vecnormalize.pkl
```

学習結果:

| 項目 | 結果 |
| --- | ---: |
| 継続後累計step | 200,704 |
| 今回追加step | 100,352 |
| 実測学習時間 | 457.792秒 |
| steps/sec | 219.209 |

TensorBoardログ:

```text
runs\tensorboard\phase25_vecnorm_continue_100k_seed4242_0
```

評価条件は前回と同じく、評価seed列`6000`から`6019`、20エピソード、最大1000 step、deterministic、学習環境とは別の評価環境です。VecNormalize評価では保存済み統計を読み込み、統計更新なし、報酬正規化なしにしています。

評価コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_vecnormalize.pkl --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --csv work\phase25_vecnorm_continue_100k_eval.csv
```

100,000ステップ時点との比較:

| モデル | 平均報酬 | 平均生存step | 平均前方移動距離 | 転倒率 |
| --- | ---: | ---: | ---: | ---: |
| VecNormalize 100k | 370.75 | 68.80 | 0.794 | 100% |
| VecNormalize 継続後 200k | 549.04 | 109.10 | 0.728 | 100% |

報酬成分の比較:

| モデル | survive | forward | ctrl | contact | x_velocity | y_velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VecNormalize 100k | 339.00 | 55.28 | -12.39 | -11.13 | 44.22 | -7.71 |
| VecNormalize 継続後 200k | 540.50 | 45.73 | -21.13 | -16.06 | 36.58 | -14.66 |

今回の継続学習では、平均生存stepが`68.80`から`109.10`へ伸び、平均報酬も増えました。一方で、平均前方移動距離は`0.794`から`0.728`へ少し低下し、forward報酬とx_velocityも下がっています。つまり「長く倒れにくくなった」方向の改善はありますが、「より前へ進む」改善は今回の100,000ステップでは確認できませんでした。転倒率は引き続き100%なので、安定歩行にはまだ到達していません。

3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_vecnormalize.pkl --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

Codex上では3Dウィンドウの目視確認は行っていません。代わりに、継続後モデルで`--render-mode rgb_array --max-steps 5`を実行し、`(480, 480, 3)`の描画フレーム取得まで確認しました。300,000ステップや500,000ステップの追加学習は開始していません。

## フェーズ2.6: 200kモデル分析と次の改善方針

今回は分析と評価機能の実装のみ行い、追加学習は開始していません。既存の200kモデルとVecNormalize統計は上書きせず維持しています。標準`Humanoid-v5`の報酬関数も変更していません。

追加した機能:

- `evaluate.py`で報酬成分の合計と1 step平均を出力
- 平均前方速度、胴体高さ、胴体の上向き姿勢、左右移動量を評価
- 評価CSVへ上記メトリクスを保存
- `play.py --video`で`rgb_array`フレームから動画ファイルを保存

動画保存コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase25_vecnorm_continue_100k\phase25_vecnorm_continue_100k_seed4242_vecnormalize.pkl --render-mode rgb_array --episodes 1 --max-steps 160 --seed 6000 --deterministic --device cpu --video outputs\phase26_200k_seed6000.gif --video-fps 30
```

保存結果:

```text
outputs\phase26_200k_seed6000.gif
```

同じseed列`6000`から`6019`、20エピソード、最大1000 step、deterministicで100kモデルと200kモデルを再評価しました。

| モデル | 平均報酬 | 平均生存step | 平均前方移動距離 | 平均前方速度 | 左右移動量 | 転倒率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VecNormalize 100k | 370.75 | 68.80 | 0.794 | 0.644 | 0.267 | 100% |
| VecNormalize 200k | 549.04 | 109.10 | 0.728 | 0.352 | 0.505 | 100% |

報酬成分の合計平均:

| モデル | survive | forward | ctrl | contact |
| --- | ---: | ---: | ---: | ---: |
| VecNormalize 100k | 339.00 | 55.28 | -12.39 | -11.13 |
| VecNormalize 200k | 540.50 | 45.73 | -21.13 | -16.06 |

報酬成分の1 step平均:

| モデル | survive/step | forward/step | ctrl/step | contact/step |
| --- | ---: | ---: | ---: | ---: |
| VecNormalize 100k | 4.927 | 0.804 | -0.180 | -0.163 |
| VecNormalize 200k | 4.953 | 0.439 | -0.194 | -0.148 |

姿勢・高さ:

| モデル | 平均胴体高さ | 最低胴体高さ平均 | 平均胴体上向き | 最低胴体上向き平均 |
| --- | ---: | ---: | ---: | ---: |
| VecNormalize 100k | 1.191 | 1.012 | 0.876 | 0.739 |
| VecNormalize 200k | 1.205 | 1.011 | 0.788 | 0.618 |

VecNormalize統計:

| 項目 | 100k | 200k |
| --- | ---: | ---: |
| obs平均の絶対値平均 | 2.187 | 2.299 |
| obs分散平均 | 1119.524 | 1027.820 |
| return平均 | 134.924 | 153.564 |
| return分散 | 6972.022 | 8034.731 |

PPO設定は100kと200kで同じです。

| 項目 | 値 |
| --- | --- |
| `n_steps` | 512 |
| `batch_size` | 64 |
| `n_epochs` | 5 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| optimizer状態 | 100k/200kとも非空、13 entries |

計測値として、200kモデルは100kモデルより長く生存しますが、前方速度とforward報酬は下がり、左右移動量が増え、胴体の上向き姿勢も悪化しています。解釈としては、標準報酬の生存報酬が強く、PPOが「前へ速く歩く」よりも「横に流れながら長く倒れない」挙動を選び始めている可能性が高いです。3D画面では歩行に近い動きが見られているため、運動パターンの芽はありますが、評価値では安定した前進歩行には未到達です。

学習速度については、VecNormalize新規100k実験では`559.118 steps/sec`、継続100k実験では`219.209 steps/sec`でした。PPOハイパーパラメータとモデル構造は同じなので、設定変更そのものが原因とは考えにくいです。候補としては、長時間実行後のCPU熱・電力制限、バックグラウンド負荷、Windows側の省電力状態、または継続学習時の実行タイミング差が考えられます。次回長めの学習前に、同じ起動直後条件で短いベンチマークを再計測するのが安全です。

次の優先方針:

1. 報酬設計を最優先にする。標準`Humanoid-v5`を直接変更せず、別名のラッパー環境として前方速度、左右移動ペナルティ、胴体姿勢、転倒ペナルティを明示した実験環境を作るのが良いです。
2. 次にPPO設定を調整する。`n_steps=2048`、`batch_size=256`、`n_epochs=10`、学習率`1e-4`から`3e-4`程度の比較を候補にします。
3. カリキュラム学習はその後に検討します。まずは「前に進むほど得、横流れと倒れ込みは損」という評価軸を明確にしたほうが、カリキュラムの効果も判断しやすくなります。

したがって、次の実験は「標準Humanoid-v5は保存したまま、報酬ラッパー環境を追加して短い対照実験を行う」方針を推奨します。

## フェーズ2.7: 安定前進用の新報酬ラッパー

今回は実装と短い動作確認のみ行い、長時間学習は開始していません。既存の100k/200kモデルとVecNormalize統計は上書きしていません。標準`Humanoid-v5`の報酬関数も変更していません。

追加ファイル:

```text
humanoid_ai\phase27_reward.py
configs\phase27_reward_default.json
```

既存スクリプトへの追加:

- `train_phase25.py --reward-config configs\phase27_reward_default.json`
- `evaluate.py --reward-config configs\phase27_reward_default.json`
- `play.py --reward-config configs\phase27_reward_default.json`

座標・姿勢の取得方法:

| 指標 | 取得元 | 意味 |
| --- | --- | --- |
| 前方速度 | `info["x_velocity"]` | MuJoCo HumanoidのワールドX方向速度 |
| 横方向速度 | `info["y_velocity"]` | MuJoCo HumanoidのワールドY方向速度 |
| 胴体高さ | `env.unwrapped.data.qpos[2]` | torso rootのZ高さ |
| 胴体姿勢 | `env.unwrapped.data.xmat[torso_body_id][8]` | torsoローカルZ軸とワールドZ軸の一致度 |

標準`Humanoid-v5`の報酬は次の構造です。

```text
standard_reward = reward_survive + reward_forward + reward_ctrl + reward_contact
```

フェーズ2.7の新報酬では、標準の`reward_forward`は合計に入れず、目標前進速度への追従報酬に置き換えます。これにより前進報酬の二重計上を避けます。`reward_survive`、`reward_ctrl`、`reward_contact`は1回だけ使用します。

新報酬の式:

```text
velocity_tracking = forward_tracking_weight * exp(-((x_velocity - target_forward_velocity)^2) / (2 * velocity_sigma^2))
lateral_penalty = -lateral_velocity_weight * y_velocity^2
tilt_penalty = -torso_tilt_weight * (1 - torso_upright)^2
fall_penalty_term = fall_penalty if terminated else 0

phase27_reward =
    survive_weight * reward_survive
  + velocity_tracking
  + ctrl_weight * reward_ctrl
  + contact_weight * reward_contact
  + lateral_penalty
  + tilt_penalty
  + fall_penalty_term
```

デフォルト重み:

| 項目 | 値 |
| --- | ---: |
| `target_forward_velocity` | 1.0 |
| `velocity_sigma` | 0.5 |
| `forward_tracking_weight` | 3.0 |
| `survive_weight` | 1.0 |
| `ctrl_weight` | 1.0 |
| `contact_weight` | 1.0 |
| `lateral_velocity_weight` | 1.0 |
| `torso_tilt_weight` | 2.0 |
| `fall_penalty` | -25.0 |

報酬成分の単体確認:

```text
reward = 5.350327940560476
component_sum = 5.350327940560476
diff = 0.0
standard reward_forward was present in info but not included in phase27_reward
```

1024 step smoke test:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 1024 --seed 4242 --run-name phase27_reward_smoke_seed4242 --checkpoint-dir work\phase27_reward_smoke --final-name-prefix phase27_reward_smoke_seed4242 --log-dir work\phase27_tensorboard --save-freq 512 --device cpu --vec-normalize --reward-config configs\phase27_reward_default.json
```

結果:

| 項目 | 結果 |
| --- | ---: |
| 学習step | 1024 |
| 学習時間 | 2.160秒 |
| steps/sec | 474.030 |
| 保存モデル | `work\phase27_reward_smoke\phase27_reward_smoke_seed4242_latest.zip` |
| VecNormalize統計 | `work\phase27_reward_smoke\phase27_reward_smoke_seed4242_vecnormalize.pkl` |

TensorBoardには`phase27/phase27_reward_total`、`phase27/phase27_reward_velocity_tracking`、`phase27/phase27_penalty_lateral_velocity`、`phase27/phase27_penalty_torso_tilt`などを記録できることを確認しました。Monitor CSVにも同じ報酬成分列が出力されています。

smoke model評価:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model work\phase27_reward_smoke\phase27_reward_smoke_seed4242_latest.zip --vec-normalize-stats work\phase27_reward_smoke\phase27_reward_smoke_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json --episodes 3 --max-steps 200 --seed 6000 --deterministic --device cpu --csv work\phase27_reward_smoke_eval.csv
```

評価は3エピソードで完了し、CSVに標準の物理指標と新報酬成分を保存できました。1024 stepは動作確認用なので、歩行性能の良否判断には使いません。

比較実験の準備:

- 標準報酬PPO: `--reward-config`なし
- 新報酬PPO: `--reward-config configs\phase27_reward_default.json`
- どちらも新規PPOモデル、VecNormalizeあり、同じseed、同じPPOハイパーパラメータで比較可能です。
- 評価は報酬総量ではなく、前進距離、前進速度、生存step、転倒率、左右移動量、胴体姿勢で比較します。

推定所要時間:

| 実験 | 100,000 step推定 |
| --- | ---: |
| 標準報酬PPO VecNormalizeあり | 約3分から5分 |
| 新報酬PPO VecNormalizeあり | 約4分から8分 |
| 2本合計 | 約7分から13分 |

上記は今回の1024 step smoke testと過去の100k実測からの概算です。CPUの発熱やWindowsの電力制限で遅くなる可能性があるため、本格比較前に5,000 step程度の速度計測を挟むのが安全です。

## フェーズ2.8: 標準報酬と新報酬の対照実験

フェーズ2.8では、標準`Humanoid-v5`報酬とフェーズ2.7の新報酬を、どちらも`PPO + VecNormalize`、新規モデル、`seed=4242`、CPU実行、描画なし、100,000 stepで比較しました。既存の100k/200kモデルとVecNormalize統計は上書きしていません。

最初に5,000 step程度の速度計測を行いました。実際にはPPOのロールアウト単位により5,120 step実行されています。

| 条件 | 実測step | 学習時間 | steps/sec | 100k推定 |
| --- | ---: | ---: | ---: | ---: |
| 標準報酬 + VecNormalize | 5,120 | 11.133秒 | 459.910 | 約217秒 |
| 新報酬 + VecNormalize | 5,120 | 11.182秒 | 457.896 | 約218秒 |

2本合計の推定は約7.3分で15分以内だったため、承認条件どおり続けて対照実験を実行しました。

実行コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase28_standard_100k_seed4242 --checkpoint-dir checkpoints\phase28_standard_100k --final-name-prefix phase28_standard_100k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize

.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase28_reward_100k_seed4242 --checkpoint-dir checkpoints\phase28_reward_100k --final-name-prefix phase28_reward_100k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --reward-config configs\phase27_reward_default.json
```

実測学習時間:

| 条件 | 実学習step | 学習時間 | steps/sec | 保存先 |
| --- | ---: | ---: | ---: | --- |
| 標準報酬 + VecNormalize | 100,352 | 183.109秒 | 548.046 | `checkpoints\phase28_standard_100k` |
| 新報酬 + VecNormalize | 100,352 | 184.070秒 | 545.184 | `checkpoints\phase28_reward_100k` |

保存ファイル:

```text
checkpoints\phase28_standard_100k\phase28_standard_100k_seed4242_latest.zip
checkpoints\phase28_standard_100k\phase28_standard_100k_seed4242_vecnormalize.pkl
checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_latest.zip
checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_vecnormalize.pkl
```

評価は両モデルとも同じseed列`6000`から20エピソード、決定論的行動、VecNormalize統計は読み込み後に統計更新と報酬正規化を無効にして実施しました。

評価コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase28_standard_100k\phase28_standard_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase28_standard_100k\phase28_standard_100k_seed4242_vecnormalize.pkl --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --target-forward-velocity 1.0 --target-velocity-tolerance 0.25 --csv work\phase28_standard_100k_eval.csv

.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --target-forward-velocity 1.0 --target-velocity-tolerance 0.25 --csv work\phase28_reward_100k_eval.csv
```

評価結果:

| 指標 | 標準報酬 | 新報酬 | 見方 |
| --- | ---: | ---: | --- |
| 平均生存step | 68.80 | 75.55 | 新報酬が改善 |
| 平均前方移動距離 | 0.794 | 0.840 | 新報酬が小幅改善 |
| 平均前進速度 | 0.644 | 0.520 | 標準報酬が高い |
| 転倒率 | 100% | 100% | 改善なし |
| 平均左右移動量 | 0.267 | 0.233 | 新報酬が改善 |
| 平均胴体高さ | 1.191 | 1.232 | 新報酬が改善 |
| 平均胴体姿勢 | 0.876 | 0.932 | 新報酬が改善 |
| 平均最小胴体姿勢 | 0.739 | 0.797 | 新報酬が改善 |
| 目標速度1.0±0.25内の割合 | 42.5% | 25.2% | 標準報酬が高い |

報酬成分の1 step平均:

| 成分 | 標準報酬 | 新報酬 |
| --- | ---: | ---: |
| 標準`reward_survive` | 4.927 | 4.933 |
| 標準`reward_forward` | 0.804 | 0.650 |
| 標準`reward_ctrl` | -0.180 | -0.189 |
| 標準`reward_contact` | -0.163 | -0.125 |
| 新報酬`velocity_tracking` | 0.000 | 1.635 |
| 新報酬`lateral_penalty` | 0.000 | -0.027 |
| 新報酬`tilt_penalty` | 0.000 | -0.018 |
| 新報酬`fall_penalty` | 0.000 | -0.335 |

CSV:

```text
work\phase28_standard_100k_eval.csv
work\phase28_reward_100k_eval.csv
work\phase28_comparison_summary.csv
```

TensorBoard:

```powershell
.\.venv_phase1\Scripts\tensorboard.exe --logdir runs\tensorboard
```

再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase28_standard_100k\phase28_standard_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase28_standard_100k\phase28_standard_100k_seed4242_vecnormalize.pkl --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

分析:

- 新報酬は、100k時点では生存時間、前方移動距離、左右移動量、胴体高さ、胴体姿勢を改善しました。
- 一方で平均前進速度と目標速度付近にいた割合は標準報酬より低く、速度追従よりも姿勢維持寄りの挙動になっています。
- 転倒率はどちらも100%で、安定歩行が完成したとは言えません。
- 異なる報酬関数なので、総報酬は性能比較の根拠にしていません。比較には前進距離、前進速度、生存step、転倒率、左右移動量、胴体姿勢を使います。
- 100kだけでは初期探索の影響が大きいため、結論は暫定です。次は新報酬の速度追従重みや転倒ペナルティを調整するか、同じ報酬で300k程度まで伸ばして傾向を見るのが候補です。ただし追加の長時間学習は未実行です。

確認できたこと:

- CPUで標準報酬・新報酬の100k学習が完了しました。
- 学習中の3D描画は無効でした。
- モデルとVecNormalize統計はそれぞれ別フォルダに保存しました。
- 評価時は保存済みVecNormalize統計を読み込み、統計更新と報酬正規化を無効にしました。
- 20エピソード評価とCSV保存が完了しました。

未確認のこと:

- `--render-mode human`の3Dウィンドウ表示は、この実験中には自動確認していません。上記の再生コマンドで目視確認してください。

## フェーズ2.9: 新報酬モデルの追加100k継続学習

フェーズ2.9では、フェーズ2.8の新報酬`PPO + VecNormalize` 100kモデルから、同じ報酬設定`configs\phase27_reward_default.json`を変更せずに追加100,000 stepだけ継続学習しました。既存の100kモデル、VecNormalize統計、ログは上書きしていません。

読み込み元:

```text
checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_latest.zip
checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_vecnormalize.pkl
```

保存先:

```text
checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_latest.zip
checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_steps_200704.zip
checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_vecnormalize.pkl
```

継続学習開始時の確認:

| 項目 | 値 |
| --- | ---: |
| `model_start_num_timesteps` | 100,352 |
| `optimizer_state_entries` | 13 |
| `vecnormalize_training` | True |
| `vecnormalize_norm_reward` | True |

上記により、PPOの学習stepとoptimizer状態を引き継ぎ、VecNormalize統計も学習時に更新されていることを確認しました。

学習コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase29_reward_200k_seed4242 --checkpoint-dir checkpoints\phase29_reward_200k --final-name-prefix phase29_reward_200k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --resume checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase28_reward_100k\phase28_reward_100k_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json
```

学習結果:

| 項目 | 結果 |
| --- | ---: |
| 累計step | 200,704 |
| 今回追加step | 100,352 |
| 学習時間 | 183.376秒 |
| steps/sec | 547.247 |

フェーズ2.8の新報酬100k実験は545.184 steps/secだったため、今回の実測では学習速度低下は見られませんでした。学習中の3D描画は無効です。

評価は100k時点と同じseed列`6000`から20エピソード、決定論的行動、同じ新報酬環境、保存済みVecNormalize統計を読み込み、統計更新と報酬正規化を無効にして実施しました。

評価コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --target-forward-velocity 1.0 --target-velocity-tolerance 0.25 --csv work\phase29_reward_200k_eval.csv
```

100kとの比較:

| 指標 | 100k | 200k | 変化 |
| --- | ---: | ---: | --- |
| 平均生存step | 75.55 | 99.25 | 改善 |
| 平均前方移動距離 | 0.840 | 0.632 | 悪化 |
| 平均前進速度 | 0.520 | 0.327 | 悪化 |
| 転倒率 | 100% | 100% | 改善なし |
| 平均左右移動量 | 0.233 | 0.495 | 悪化 |
| 平均胴体高さ | 1.232 | 1.237 | ほぼ同等 |
| 平均胴体姿勢 | 0.932 | 0.890 | 悪化 |
| 平均最小胴体姿勢 | 0.797 | 0.729 | 悪化 |
| 目標速度1.0±0.25内の割合 | 25.2% | 10.8% | 悪化 |

解釈:

- 追加100kで平均生存stepは伸びましたが、前方移動距離と前進速度は低下しました。
- 左右移動量が大きく増え、姿勢指標も悪化したため、横方向へ流れながら生存報酬を得る傾向が強くなった可能性があります。
- 転倒率は20/20で100%のままです。報酬が増えていても、安定歩行に成功したとは判断できません。
- 現在の新報酬は、生存時間を伸ばす一方で、横流れや低速化を十分に抑えられていません。次に進む場合は、追加学習を伸ばす前に報酬重みの再調整、特に横速度ペナルティ、目標速度追従、転倒ペナルティのバランスを見直すのが安全です。

CSV:

```text
work\phase29_reward_200k_eval.csv
work\phase29_reward_100k_vs_200k_summary.csv
```

TensorBoard:

```powershell
.\.venv_phase1\Scripts\tensorboard.exe --logdir runs\tensorboard
```

GIF保存:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json --render-mode rgb_array --video outputs\phase29_reward_200k_seed6000.gif --video-fps 30 --episodes 1 --max-steps 160 --seed 6000 --deterministic --device cpu
```

GIF保存は成功し、`last_rgb_frame_shape=(480, 480, 3)`を確認しました。エピソード1は117 stepで転倒終了しました。

3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase29_reward_200k\phase29_reward_200k_seed4242_vecnormalize.pkl --reward-config configs\phase27_reward_default.json --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

確認できたこと:

- 100kモデルとVecNormalize統計を読み込んで継続学習しました。
- PPO optimizer状態と学習stepが継続されていることをログで確認しました。
- VecNormalize統計は学習時に更新されました。
- 追加100kのみ実施し、300k以上の学習は開始していません。
- 同じseed列で20エピソード評価し、CSVとREADMEへ記録しました。
- `rgb_array`でGIFを保存できました。

未確認のこと:

- `--render-mode human`の3Dウィンドウ表示は、この実験中には自動確認していません。上記の3D再生コマンドで目視確認してください。

## フェーズ3: カリキュラム学習の実装と短い動作確認

フェーズ2.9の結果では、新報酬モデルを100kから200kへ継続学習すると平均生存stepは`75.55 -> 99.25`へ伸びましたが、平均前方移動距離は`0.840 -> 0.632`、平均前進速度は`0.520 -> 0.327`へ悪化しました。左右移動量は`0.233 -> 0.495`へ増え、転倒率は`100%`のままでした。

原因調査用に以下を追加しました。

```text
analyze_phase3_cause.py
work\phase3_cause_analysis.json
work\phase3_cause_analysis_summary.csv
```

調査結果:

| 指標 | 100k | 200k | 変化 |
| --- | ---: | ---: | ---: |
| 平均報酬 | 444.260 | 540.449 | +96.189 |
| 平均生存step | 75.550 | 99.250 | +23.700 |
| 平均前方移動距離 | 0.840 | 0.632 | -0.208 |
| 平均前進速度 | 0.520 | 0.327 | -0.193 |
| 平均左右移動量 | 0.233 | 0.495 | +0.262 |
| 平均胴体姿勢 | 0.932 | 0.890 | -0.042 |
| 目標速度内割合 | 25.2% | 10.8% | -14.3pt |
| 転倒率 | 100% | 100% | 0pt |

報酬成分の1 step平均:

| 成分 | 100k | 200k | 解釈 |
| --- | ---: | ---: | --- |
| survive | 4.933 | 4.948 | 生存中はほぼ一定 |
| velocity_tracking | 1.635 | 1.277 | 目標速度から遠ざかった |
| lateral_penalty | -0.027 | -0.130 | 横流れが悪化 |
| torso_tilt_penalty | -0.018 | -0.044 | 姿勢が悪化 |
| fall_penalty | -0.335 | -0.260 | 長く生きたため1 step平均では薄まった |

計測値からの解釈:

- 目標速度追従報酬の式は、`x_velocity`が目標速度に近いほど大きくなる形で機能しています。
- ただし、200kでは生存時間の増加によって総報酬が伸びる一方、速度追従、横方向ペナルティ、姿勢ペナルティの改善にはつながっていません。
- 現在の重みでは、学習が「前へ安定して歩く」よりも「横に流れながら少し長く生きる」方へ寄っています。
- そのため、いきなり目標速度`1.0`を狙うのではなく、まず姿勢維持と低速前進を安定させるカリキュラム学習を優先します。

追加したカリキュラム報酬:

```text
humanoid_ai\phase3_curriculum_reward.py
configs\phase3_curriculum_default.json
tests\test_phase3_curriculum_reward.py
```

報酬式:

```text
reward =
  survive_weight * reward_survive
  + forward_tracking_weight * exp(-((x_velocity - target_forward_velocity)^2) / (2 * velocity_sigma^2))
  + ctrl_weight * reward_ctrl
  + contact_weight * reward_contact
  - lateral_velocity_weight * y_velocity^2
  - torso_tilt_weight * (1 - torso_upright)^2
  - torso_height_weight * max(0, min_torso_height - torso_height)^2
  + fall_penalty_if_terminated
```

標準の`Humanoid-v5`自体は変更していません。`RewardWrapper`で別環境として扱い、`--curriculum-config`と`--curriculum-stage`を指定したときだけ有効になります。標準報酬と二重計上しないよう、標準の合計報酬は記録用`phase3_original_reward`に残し、学習には上記の新しい合計報酬`phase3_reward_total`を返します。

Stage設定:

| Stage | 目的 | 目標速度 | 主な重み |
| --- | --- | ---: | --- |
| stage1 | 姿勢維持とごく低速の前進 | 0.25 | 姿勢・横流れペナルティ強め、生存報酬控えめ |
| stage2 | 低速の前進 | 0.55 | 前進追従を強め、姿勢制約を維持 |
| stage3 | 目標速度での継続歩行 | 1.00 | 前進追従をさらに強める |

Stage成功条件:

| Stage | 平均生存step | 転倒率 | 平均前方距離 | 平均左右移動量 | 追加条件 |
| --- | ---: | ---: | ---: | ---: | --- |
| stage1 | 150以上 | 80%以下 | 0.35以上 | 0.35以下 | 胴体姿勢0.90以上 |
| stage2 | 250以上 | 60%以下 | 1.20以上 | 0.50以下 | 前進速度0.35以上、胴体姿勢0.85以上 |
| stage3 | 500以上 | 30%以下 | 4.00以上 | 0.80以下 | 前進速度0.75以上、胴体姿勢0.80以上 |

単体テスト:

```powershell
.\.venv_phase1\Scripts\python.exe tests\test_phase3_curriculum_reward.py
```

結果:

```text
phase3_curriculum_reward_tests_done=true
```

Stage 1の1024 step smoke test:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 1024 --seed 4242 --run-name phase3_stage1_smoke_seed4242 --checkpoint-dir work\phase3_stage1_smoke --final-name-prefix phase3_stage1_smoke_seed4242 --log-dir work\phase3_tensorboard --save-freq 512 --device cpu --vec-normalize --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1
```

結果:

| 項目 | 値 |
| --- | ---: |
| 学習step | 1024 |
| 学習時間 | 2.238秒 |
| steps/sec | 457.529 |
| 保存モデル | `work\phase3_stage1_smoke\phase3_stage1_smoke_seed4242_latest.zip` |
| VecNormalize統計 | `work\phase3_stage1_smoke\phase3_stage1_smoke_seed4242_vecnormalize.pkl` |

保存済みVecNormalize統計を読み込んだ短い評価:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model work\phase3_stage1_smoke\phase3_stage1_smoke_seed4242_latest.zip --vec-normalize-stats work\phase3_stage1_smoke\phase3_stage1_smoke_seed4242_vecnormalize.pkl --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1 --episodes 3 --max-steps 200 --seed 6000 --deterministic --device cpu --target-forward-velocity 0.25 --target-velocity-tolerance 0.15 --csv work\phase3_stage1_smoke_eval.csv
```

評価結果:

| 指標 | 値 |
| --- | ---: |
| 平均報酬 | 109.748 |
| 平均生存step | 42.667 |
| 平均前方移動距離 | 0.168 |
| 平均前進速度 | 0.037 |
| 平均左右移動量 | 0.150 |
| 平均胴体姿勢 | 0.892 |
| 目標速度内割合 | 11.0% |
| 転倒率 | 100% |

この評価は1024 stepのsmoke testであり、歩行習得を示すものではありません。目的は、報酬計算、VecNormalize保存・読み込み、評価CSV出力が動くことの確認です。

Stage移行の短い確認:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 512 --seed 4242 --run-name phase3_stage2_resume_smoke_seed4242 --checkpoint-dir work\phase3_stage2_resume_smoke --final-name-prefix phase3_stage2_resume_smoke_seed4242 --log-dir work\phase3_tensorboard --save-freq 512 --device cpu --vec-normalize --resume work\phase3_stage1_smoke\phase3_stage1_smoke_seed4242_latest.zip --vec-normalize-stats work\phase3_stage1_smoke\phase3_stage1_smoke_seed4242_vecnormalize.pkl --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage2
```

結果:

```text
model_start_num_timesteps=1024
optimizer_state_entries=13
phase3_stage_index=2
trained_this_run=512
```

同一PPOモデルを読み込み、optimizer状態と学習stepを維持したままStage 2報酬へ移行できることを確認しました。保存先は`work\phase3_stage2_resume_smoke`で、既存の学習済みモデル・統計・ログは上書きしていません。

TensorBoard:

```powershell
.\.venv_phase1\Scripts\tensorboard.exe --logdir work\phase3_tensorboard
```

次の本格カリキュラム学習の推定時間:

| 条件 | 推定時間 |
| --- | ---: |
| 100,000 step / 1 stage | 約3.1〜3.6分 |
| 100,000 step x 3 stages | 約9.2〜10.9分 |
| 300,000 step / 1 stage | 約9.2〜10.9分 |
| 300,000 step x 3 stages | 約27.5〜32.8分 |
| 500,000 step / 1 stage | 約15.3〜18.2分 |

本格学習の推奨順序:

1. `stage1`を100,000 step学習し、姿勢・横流れ・転倒率が改善するか確認する。
2. 成功条件に近づいた場合のみ、同一モデルを`stage2`へ移行する。
3. `stage2`で前進距離と速度が改善した場合のみ、`stage3`へ移行する。

まだ本格的なカリキュラム学習は開始していません。開始する場合は、最初に`stage1`の100,000 step実験を承認してください。

## フェーズ3: Stage 1 100,000 step実験

Stage 1の100,000 step学習を新規PPOモデルから実行しました。smoke testモデルからの継続ではありません。標準`Humanoid-v5`は変更せず、`configs\phase3_curriculum_default.json`の`stage1`設定、PPO + VecNormalize、seed=`4242`、CPU、学習時3D描画なしで実施しました。

学習コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase3_stage1_100k_seed4242 --checkpoint-dir checkpoints\phase3_stage1_100k --final-name-prefix phase3_stage1_100k_seed4242 --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1
```

学習結果:

| 項目 | 値 |
| --- | ---: |
| 指定step | 100,000 |
| 実学習step | 100,352 |
| 学習時間 | 179.978秒 |
| steps/sec | 557.579 |
| PyTorch threads | 10 |
| CUDA | 利用不可 |

PPOはrollout単位で更新するため、実際の保存stepは`100,352`になりました。

保存先:

```text
checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_latest.zip
checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_steps_100352.zip
checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_vecnormalize.pkl
```

TensorBoard:

```powershell
.\.venv_phase1\Scripts\tensorboard.exe --logdir runs\tensorboard
```

該当ログ:

```text
runs\tensorboard\phase3_stage1_100k_seed4242_1
```

評価コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe evaluate.py --agent model --model checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1 --episodes 20 --max-steps 1000 --seed 6000 --deterministic --device cpu --target-forward-velocity 0.25 --target-velocity-tolerance 0.15 --csv work\phase3_stage1_100k_eval.csv
```

評価条件:

| 項目 | 値 |
| --- | ---: |
| 評価エピソード数 | 20 |
| エピソード最大長 | 1000 step |
| 評価seed開始値 | 6000 |
| terminated | 20 |
| truncated | 0 |
| VecNormalize統計更新 | 無効 |
| 報酬正規化 | 無効 |

評価結果:

| 指標 | 結果 | Stage 1条件 | 判定 |
| --- | ---: | ---: | --- |
| 平均生存step | 82.60 | 150以上 | 未達 |
| 転倒率 | 100% | 80%以下 | 未達 |
| 平均前進距離 | 0.729 | 0.35以上 | 達成 |
| 平均前進速度 | 0.417 | 参考値 | - |
| 平均左右移動量 | 0.233 | 0.35以下 | 達成 |
| 平均胴体姿勢 | 0.971 | 0.90以上 | 達成 |
| 目標速度0.25±0.15内割合 | 49.1% | 参考値 | - |

報酬成分の1 step平均:

| 成分 | 1 step平均 |
| --- | ---: |
| phase3_reward_total | 3.094 |
| survive | 2.963 |
| velocity_tracking | 0.962 |
| ctrl | -0.140 |
| contact | -0.112 |
| lateral_velocity_penalty | -0.133 |
| torso_tilt_penalty | -0.016 |
| torso_height_penalty | -0.000 |
| fall_penalty | -0.429 |

Stage 1合否:

```text
不合格
```

前進距離、左右移動量、胴体姿勢は条件を満たしましたが、平均生存stepと転倒率が未達です。20エピソードすべてが`terminated=True`で終了し、`truncated=True`は0回でした。100k時点では、低速前進と姿勢維持の兆候はありますが、安定して立ち続ける段階には到達していません。

GIF保存:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1 --render-mode rgb_array --video outputs\phase3_stage1_100k_seed6000.gif --video-fps 30 --episodes 1 --max-steps 160 --seed 6000 --deterministic --device cpu
```

結果:

```text
episode=1 reward=289.663 length=90 terminated=True truncated=False
last_rgb_frame_shape=(480, 480, 3)
```

3Dウィンドウ再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

原因分析:

- Stage 1の報酬は前進距離、横流れ抑制、姿勢維持には一定の効果がありました。
- ただし、転倒時ペナルティが総episode報酬に対して十分に強くなく、80〜90 step程度で倒れても報酬上はある程度許容される状態です。
- 目標速度`0.25`に対し、評価時の平均前進速度は`0.417`で、少し速すぎます。速度追従は効いていますが、低速で安定するより前へ進む行動を選んでいる可能性があります。
- 平均胴体姿勢は`0.971`で良好ですが、最終的には全エピソードで転倒しているため、倒れる直前の局所的な姿勢崩れを十分に抑えられていません。

次の改善案:

1. Stage 1の`fall_penalty`を強める。
2. `survive_weight`を少し上げるか、一定step以上生存した場合のボーナスを追加する。
3. `target_forward_velocity`を`0.25`から`0.15〜0.20`へ下げ、まず転倒率改善を優先する。
4. 胴体高さまたは最小胴体姿勢へのペナルティを強め、倒れる直前の姿勢崩れを早めに罰する。

今回はStage 1の100,000 stepのみ実行し、Stage 2・Stage 3は開始していません。

## フェーズ3.1: Stage 1転倒原因分析と安定化実験の準備

Stage 1 100kモデルを維持したまま、同一seed列`6000`から20エピソードの詳細ステップログを取得しました。保存済みVecNormalize統計で観測を正規化し、物理状態は通常のGym環境から取得しています。既存モデル、VecNormalize統計、ログ、設定は上書きしていません。

追加した解析スクリプト:

```text
analyze_phase31_falls.py
```

実行コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe analyze_phase31_falls.py --model checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase3_stage1_100k\phase3_stage1_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase3_curriculum_default.json --curriculum-stage stage1 --episodes 20 --max-steps 1000 --seed 6000 --device cpu --window-steps 20 --step-csv work\phase31_stage1_100k_step_log.csv --fall-window-csv work\phase31_stage1_100k_fall_windows.csv --analysis-json work\phase31_stage1_100k_fall_analysis.json --compare-csv work\phase31_standard_vs_stage1_physics.csv --trajectory-svg outputs\phase31_stage1_trajectories.svg --height-svg outputs\phase31_stage1_torso_height.svg
```

生成物:

```text
work\phase31_stage1_100k_step_log.csv
work\phase31_stage1_100k_fall_windows.csv
work\phase31_stage1_100k_fall_analysis.json
work\phase31_standard_vs_stage1_physics.csv
outputs\phase31_stage1_trajectories.svg
outputs\phase31_stage1_torso_height.svg
```

記録した主な項目:

- 胴体位置、胴体高さ、胴体姿勢、傾き角度
- 前後速度、左右速度
- 全関節角度、全関節速度
- 全actuatorの制御入力
- 報酬成分
- 左右足の床接地、両足接地、足以外の床接地

転倒直前20 stepの集計:

| 指標 | 平均 |
| --- | ---: |
| 詳細ログ上の平均episode長 | 83.30 |
| 最終胴体高さ | 0.984 |
| 直前20 stepの胴体高さ低下 | 0.250 |
| 最終胴体姿勢 | 0.909 |
| 直前20 stepの最小胴体姿勢 | 0.895 |
| 最大傾き角度 | 24.71度 |
| 直前20 step平均前進速度 | 1.017 |
| 直前20 step平均左右速度 | -0.266 |
| 直前20 step最大左右速度絶対値 | 0.357 |
| 直前20 step最大関節速度絶対値 | 17.666 |
| 左足接地率 | 16.5% |
| 右足接地率 | 73.3% |
| 両足接地率 | 9.5% |
| 足以外の床接地率 | 0.0% |

計測で確認できた事実:

- 転倒直前20 stepで胴体高さが平均`0.250`低下しています。
- 姿勢は平均では大きく崩れていませんが、最小胴体姿勢は`0.895`、最大傾き角度は平均`24.71度`まで悪化します。
- 直前20 stepの平均前進速度は`1.017`で、Stage 1の目標速度`0.25`よりかなり速くなっています。
- 右足接地率が`73.3%`、左足接地率が`16.5%`で、接地が右足に偏っています。
- 足以外の床接地は平均`0.0%`で、評価した範囲では胴体や手が床に触れてから倒れているのではなく、高さ条件などで終了している可能性が高いです。
- 制御入力は多くのstepで上限`0.4`に達しており、関節速度も最大平均`17.666`まで上がっています。

仮説:

- Stage 1モデルは、姿勢を比較的保ったまま前に進めていますが、目標速度より速く前へ出すぎて、最後に高さが落ちて終了している可能性があります。
- 右足接地に偏っているため、左右交互の安定歩行というより、片足支持に寄った倒れ込みに近い動きになっている可能性があります。
- 転倒直前に関節速度と制御入力が大きく、姿勢回復のための過補正が安定性を崩している可能性があります。

標準報酬100kモデルとの共通物理指標比較:

| 指標 | 標準報酬100k | Phase3 Stage1 100k |
| --- | ---: | ---: |
| 平均生存step | 68.80 | 82.60 |
| 転倒率 | 100% | 100% |
| 平均前進距離 | 0.794 | 0.729 |
| 平均前進速度 | 0.644 | 0.417 |
| 平均左右移動量 | 0.267 | 0.233 |
| 平均胴体姿勢 | 0.876 | 0.971 |
| 平均胴体高さ | 1.191 | 1.248 |

解釈:

- Stage 1報酬は標準報酬より、姿勢・高さ・横流れ・生存stepを改善しています。
- 一方で、転倒率はどちらも`100%`であり、安定化はまだ不十分です。
- 前進速度はStage 1で下がっていますが、それでも転倒直前は目標速度を大きく超えています。

報酬設定案:

| 案 | 設定ファイル | 目的 |
| --- | --- | --- |
| A | `configs\phase31_stage1_A_current.json` | 現行Stage 1の対照条件 |
| B | `configs\phase31_stage1_B_fall_penalty.json` | 転倒ペナルティを`-35 -> -70`へ強化し、生存を重視 |
| C | `configs\phase31_stage1_C_slow_stability.json` | 目標速度を`0.18`へ下げ、姿勢・高さ・横流れ安定を重視 |

報酬設定の単体テスト:

```powershell
.\.venv_phase1\Scripts\python.exe tests\test_phase31_curriculum_configs.py
```

結果:

```text
phase31_curriculum_config_tests_done=true
```

A/B/Cの1024 step smoke test:

| 案 | 学習step | 学習時間 | steps/sec | 保存先 |
| --- | ---: | ---: | ---: | --- |
| A | 1024 | 2.032秒 | 504.047 | `work\phase31_smoke_A` |
| B | 1024 | 1.849秒 | 553.720 | `work\phase31_smoke_B` |
| C | 1024 | 1.951秒 | 524.896 | `work\phase31_smoke_C` |

3案とも、PPO + VecNormalizeで新規モデルから学習でき、モデルとVecNormalize統計を保存できました。1024 stepのsmoke testは動作確認であり、報酬変更の有効性判断には使いません。

次の本格比較実験の推定時間:

| 条件 | 推定時間 |
| --- | ---: |
| 100,000 step / 1案 | 約3.0〜3.6分 |
| 100,000 step x 3案 | 約9〜11分 |
| 評価20エピソード x 3案 | 約1〜2分 |

推奨する次の実験:

1. A/B/Cをそれぞれ新規PPO + VecNormalizeで100,000 step学習する。
2. 同じseed列で20エピソード評価する。
3. 合計報酬ではなく、平均生存step、転倒率、前進距離、左右移動量、胴体姿勢、転倒直前20 stepを比較する。

今回は解析、設定作成、単体テスト、1024 step smoke testのみ実施しました。Stage 2・Stage 3および長時間学習は開始していません。

## フェーズ3.2: Stage 1 A/B/C 100,000 step比較実験

フェーズ3.2では、フェーズ3.1で作成した3種類のStage 1報酬設定を、それぞれ新規PPO + VecNormalizeモデルから100,000 step学習しました。seedはすべて`4242`、CPU実行、PPOハイパーパラメータは同一、学習時の3D描画は無効です。既存モデル、統計、ログは上書きしていません。

| 条件 | 設定 | 保存先 | 実学習step | 学習時間 | steps/sec |
| --- | --- | --- | ---: | ---: | ---: |
| A | 現行Stage 1 | `checkpoints\phase32_A_current_100k` | 100,352 | 191.922秒 | 522.879 |
| B | 転倒ペナルティー強化 | `checkpoints\phase32_B_fall_penalty_100k` | 100,352 | 183.729秒 | 546.194 |
| C | 低速・姿勢重視 | `checkpoints\phase32_C_slow_stability_100k` | 100,352 | 180.967秒 | 554.533 |

評価は3条件とも同じseed列`6000`から20エピソード、最大1000 step、deterministic、保存済みVecNormalize統計を読み込み、統計更新と報酬正規化を無効にして実施しました。

| 条件 | 平均生存step | 転倒率 | 平均前進距離 | 平均前進速度 | 平均左右移動量 | 平均胴体姿勢 | terminated/truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 82.60 | 100% | 0.729 | 0.417 | 0.233 | 0.971 | 20 / 0 |
| B | 88.40 | 100% | 0.615 | 0.292 | 0.315 | 0.895 | 20 / 0 |
| C | 98.40 | 100% | 0.038 | 0.036 | 0.426 | 0.961 | 20 / 0 |

足接地率:

| 条件 | 左足接地率 | 右足接地率 | 両足接地率 | 足以外の床接地率 |
| --- | ---: | ---: | ---: | ---: |
| A | 55.2% | 51.4% | 23.9% | 0.0% |
| B | 21.6% | 82.4% | 18.3% | 0.0% |
| C | 59.4% | 78.9% | 54.1% | 0.0% |

報酬成分の1 step平均:

| 条件 | total | survive | velocity tracking | lateral penalty | tilt penalty | height penalty | fall penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 3.094 | 2.963 | 0.962 | -0.133 | -0.016 | -0.000 | -0.429 |
| B | 2.804 | 3.212 | 0.939 | -0.161 | -0.137 | -0.000 | -0.808 |
| C | 3.109 | 3.711 | 0.620 | -0.318 | -0.033 | -0.001 | -0.617 |

評価CSVと集約ファイル:

```text
work\phase32_A_current_100k_eval.csv
work\phase32_B_fall_penalty_100k_eval.csv
work\phase32_C_slow_stability_100k_eval.csv
work\phase32_abc_summary.csv
work\phase32_abc_summary.json
```

GIF保存:

```text
outputs\phase32_A_current_100k_seed6000.gif
outputs\phase32_B_fall_penalty_100k_seed6000.gif
outputs\phase32_C_slow_stability_100k_seed6000.gif
```

3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase32_B_fall_penalty_100k\phase32_B_fall_penalty_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase32_B_fall_penalty_100k\phase32_B_fall_penalty_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase31_stage1_B_fall_penalty.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase32_C_slow_stability_100k\phase32_C_slow_stability_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase32_C_slow_stability_100k\phase32_C_slow_stability_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase31_stage1_C_slow_stability.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

分析:

- Aは前進距離、前進速度、左右移動量のバランスが最も良く、フェーズ3.1のStage 1 100kと同等の傾向です。
- Bは平均生存stepがAより伸びましたが、胴体姿勢が大きく悪化し、右足接地に強く偏りました。転倒ペナルティー強化だけでは安定歩行改善にはつながっていません。
- Cは平均生存stepが最長で両足接地率も高い一方、前進距離と前進速度がほぼ消え、左右移動量が悪化しました。低速・姿勢重視設定は、前進歩行よりもその場に近い粘りへ寄った可能性があります。
- 3条件すべて転倒率は100%で、安定歩行は未達です。報酬総量は条件ごとに設計が違うため、優劣判断には使っていません。
- 次に進むなら、Aをベースに「速度を落としすぎず、転倒直前の高さ低下と片足偏りを抑える」方向で、転倒ペナルティー単独ではなく接地バランスや胴体高さ維持を含む調整を優先するのがよさそうです。

今回はA/B/C各100,000 stepまでで停止し、Stage 2・Stage 3および300,000 stepを超える追加学習は開始していません。

## フェーズ3.3: 歩行成立を判定する解析機能

フェーズ3.3では、新規学習は行わず、フェーズ3.2のA/B/Cモデルを同じseed列`6000`から20エピソード評価し、歩行らしい動きかどうかを物理指標で解析しました。接地切り替えだけを歩行と誤認しないよう、前進距離、左右移動量、足の接地中滑り、胴体姿勢、転倒有無を組み合わせて判定します。

追加した解析スクリプト:

```text
analyze_phase33_walk.py
summarize_phase32.py
tests\test_phase33_walk_metrics.py
```

実行コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe analyze_phase33_walk.py --episodes 20 --max-steps 1000 --seed 6000 --device cpu
```

出力:

```text
work\phase33_walk\phase33_A_step_log.csv
work\phase33_walk\phase33_B_step_log.csv
work\phase33_walk\phase33_C_step_log.csv
work\phase33_walk\phase33_episode_summary_all.csv
work\phase33_walk\phase33_abc_walk_summary.csv
work\phase33_walk\phase33_walk_analysis.json
outputs\phase33_walk\phase33_A_contact_timeline.svg
outputs\phase33_walk\phase33_B_contact_timeline.svg
outputs\phase33_walk\phase33_C_contact_timeline.svg
```

主な追加指標:

- 左右足の接地状態、両足接地、足以外の床接地
- 左右単脚支持の切り替え回数
- 推定歩行周期
- 足位置と足速度
- 接地中の足滑り速度
- 胴体位置、速度、高さ、高さ変動、転倒直前20 stepの高さ低下
- action/ctrlの飽和率
- `walk_candidate`: 前進距離、接地切り替え、低い足滑り、低い横流れ、胴体姿勢を同時に満たした「歩行候補断片」
- `stable_walk_success`: `walk_candidate`に加えて、転倒せず150 step以上生存した安定歩行成功判定

結果:

| 条件 | 平均生存step | 転倒率 | 歩行候補率 | 安定歩行成功率 | 前進距離 | 接地切替回数 | 推定周期step | 足滑り速度 | action飽和率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 83.30 | 100% | 85% | 0% | 0.751 | 4.70 | 29.68 | 0.153 | 50.7% |
| B | 88.60 | 100% | 25% | 0% | 0.617 | 2.15 | 26.35 | 0.129 | 44.3% |
| C | 98.45 | 100% | 10% | 0% | 0.042 | 3.20 | 42.75 | 0.077 | 61.1% |

追加の安定性指標:

| 条件 | 左右移動量 | 胴体高さ平均 | 胴体高さ標準偏差 | 転倒直前20 step高さ低下 | 胴体姿勢平均 | 最小胴体姿勢平均 | 最大関節速度平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.244 | 1.247 | 0.076 | 0.250 | 0.972 | 0.892 | 18.411 |
| B | 0.321 | 1.233 | 0.081 | 0.208 | 0.895 | 0.615 | 14.284 |
| C | 0.425 | 1.257 | 0.063 | 0.257 | 0.961 | 0.784 | 24.300 |

解釈:

- Aは接地切り替え、前進距離、姿勢、横流れの組み合わせで最も歩行候補らしい断片が多く、次の改善の基準にする価値があります。
- ただしAも全エピソードで転倒しており、`stable_walk_success`は0%です。歩行成功とは判断しません。
- Bは転倒ペナルティー強化で生存stepはAより少し伸びましたが、胴体姿勢と横流れが悪化し、接地切り替えも減っています。
- Cは生存stepが最長で足滑りが少ない一方、前進距離がほぼ消え、action飽和率と横流れが増えています。低速・姿勢重視が「前進しない粘り」へ寄った可能性があります。
- 接地切り替えだけでは歩行判定に使えません。Cは切り替えがあっても前進距離が小さく、Aも候補断片はあるものの転倒するため、安定歩行未達です。

次の研究計画:

1. Aを基準として、複数seedで再現性を確認する。
   - 推奨seed: `4242`, `5252`, `6262`
   - A設定を各100,000 step学習し、今回と同じPhase 3.3指標で比較する。
   - 推定時間: 1本約3.0〜3.3分、3本で約9〜10分、評価と解析を含めて約11〜13分。
2. A派生の報酬改善案を作る。
   - 胴体高さ維持: 転倒直前20 stepの高さ低下を早めに罰する。
   - 接地安定性: 接地中の足滑りを弱く罰する。ただし左右交互接地そのものを直接報酬化しすぎない。
   - 制御入力の滑らかさ: action/ctrl飽和率と急激な関節速度を抑えるペナルティを追加する。
3. 報酬ハックを避ける方針。
   - 左右交互接地だけを報酬にしない。
   - 歩行候補判定には、前進距離、横流れ、足滑り、胴体姿勢、生存時間を必ず同時に使う。
   - 総報酬ではなく、物理指標で比較する。
4. 次回の比較学習案。
   - A基準seed再現性3本: 約11〜13分。
   - A派生3条件 x 3seed x 100,000 step: 学習だけで約27〜30分、評価込みで約32〜36分。

今回は解析機能の実装と既存モデルの評価のみ実施しました。新規の長時間学習、Stage 2、Stage 3は開始していません。

## フェーズ3.4: Stage 1 Aの複数seed再現性検証

フェーズ3.4では、フェーズ3.3で有望だったStage 1報酬設定Aが、学習seedを変えても再現するかを確認しました。報酬設定は`configs\phase31_stage1_A_current.json`、`curriculum-stage=stage1`で固定し、PPO + VecNormalize、CPU実行、学習時の3D描画なしで実施しました。

seed `4242`はフェーズ3.2の既存モデルを再利用し、再学習や上書きはしていません。新規学習は承認範囲のseed `5252`と`6262`のみ、それぞれ100,352 stepで停止しました。Stage 2・Stage 3は開始していません。

追加した解析スクリプト:

```text
analyze_phase34_multiseed.py
tests\test_phase34_multiseed_stats.py
```

新規学習コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 5252 --run-name phase34_A_seed5252_100k --checkpoint-dir checkpoints\phase34_A_seed5252_100k --final-name-prefix phase34_A_seed5252_100k --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1

.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 6262 --run-name phase34_A_seed6262_100k --checkpoint-dir checkpoints\phase34_A_seed6262_100k --final-name-prefix phase34_A_seed6262_100k --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1
```

学習結果:

| 学習seed | モデル | 学習step | 学習時間 | steps/sec | 備考 |
| --- | --- | ---: | ---: | ---: | --- |
| 4242 | `checkpoints\phase32_A_current_100k` | 100,352 | - | - | 既存モデルを再利用 |
| 5252 | `checkpoints\phase34_A_seed5252_100k` | 100,352 | 264.871秒 | 378.871 | 新規学習 |
| 6262 | `checkpoints\phase34_A_seed6262_100k` | 100,352 | 196.373秒 | 511.028 | 新規学習 |

評価コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe analyze_phase34_multiseed.py --episodes 20 --max-steps 1000 --seed 6000 --device cpu --curriculum-stage stage1 --out-dir work\phase34_multiseed --graph-dir outputs\phase34_multiseed
```

評価条件:

- 評価seed列: `6000`〜`6019`
- 各モデル20エピソード
- 最大1000 step
- `deterministic=True`
- VecNormalizeは保存済み統計を読み込み、`training=False`、`norm_reward=False`
- 20エピソードは各モデルの評価反復であり、20回の独立した学習実験としては扱いません。

seedごとの評価平均:

| 学習seed | 平均生存step | 転倒率 | 安定歩行成功率 | 歩行候補率 | 前進距離 | 前進速度 | 左右移動量 | 胴体姿勢 | 胴体高さstd | 接地切替回数 | 推定周期step | 足滑り速度 | action飽和率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4242 | 83.30 | 100% | 0% | 85% | 0.751 | 0.427 | 0.244 | 0.972 | 0.076 | 4.70 | 29.68 | 0.153 | 50.7% |
| 5252 | 85.05 | 100% | 0% | 0% | 0.778 | 0.412 | 0.410 | 0.955 | 0.079 | 1.60 | 4.75 | 0.118 | 39.4% |
| 6262 | 82.20 | 100% | 0% | 70% | 0.860 | 0.506 | 0.072 | 0.939 | 0.077 | 3.15 | 28.99 | 0.262 | 49.4% |

3モデル間の集約:

| 指標 | 平均 | 標準偏差 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| 平均生存step | 83.52 | 1.17 | 82.20 | 85.05 |
| 転倒率 | 100% | 0% | 100% | 100% |
| 安定歩行成功率 | 0% | 0% | 0% | 0% |
| 歩行候補率 | 51.7% | 37.0% | 0% | 85.0% |
| 前進距離 | 0.796 | 0.046 | 0.751 | 0.860 |
| 前進速度 | 0.449 | 0.041 | 0.412 | 0.506 |
| 左右移動量 | 0.242 | 0.138 | 0.072 | 0.410 |
| 胴体姿勢 | 0.955 | 0.013 | 0.939 | 0.972 |
| 接地切替回数 | 3.15 | 1.27 | 1.60 | 4.70 |
| 足滑り速度 | 0.178 | 0.061 | 0.118 | 0.262 |
| action飽和率 | 46.5% | 5.0% | 39.4% | 50.7% |

出力:

```text
work\phase34_multiseed\phase34_seed4242_step_log.csv
work\phase34_multiseed\phase34_seed5252_step_log.csv
work\phase34_multiseed\phase34_seed6262_step_log.csv
work\phase34_multiseed\phase34_episode_summary_all.csv
work\phase34_multiseed\phase34_seed_summary.csv
work\phase34_multiseed\phase34_model_metric_stats.csv
work\phase34_multiseed\phase34_multiseed_analysis.json
outputs\phase34_multiseed\phase34_seed4242_contact_timeline.svg
outputs\phase34_multiseed\phase34_seed5252_contact_timeline.svg
outputs\phase34_multiseed\phase34_seed6262_contact_timeline.svg
outputs\phase34_multiseed\phase34_multiseed_metric_bars.svg
outputs\phase34_multiseed\phase34_seed4242_seed6000.gif
outputs\phase34_multiseed\phase34_seed5252_seed6000.gif
outputs\phase34_multiseed\phase34_seed6262_seed6000.gif
```

3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase34_A_seed5252_100k\phase34_A_seed5252_100k_latest.zip --vec-normalize-stats checkpoints\phase34_A_seed5252_100k\phase34_A_seed5252_100k_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase34_A_seed6262_100k\phase34_A_seed6262_100k_latest.zip --vec-normalize-stats checkpoints\phase34_A_seed6262_100k\phase34_A_seed6262_100k_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

テスト:

```powershell
.\.venv_phase1\Scripts\python.exe -m py_compile analyze_phase34_multiseed.py analyze_phase33_walk.py
.\.venv_phase1\Scripts\python.exe tests\test_phase33_walk_metrics.py
.\.venv_phase1\Scripts\python.exe tests\test_phase34_multiseed_stats.py
```

結果:

```text
phase33_walk_metric_tests_done=true
phase34_multiseed_stats_tests_done=true
```

解釈:

- 3モデルとも100,000 step級では転倒率100%、安定歩行成功率0%で、安定歩行は未達です。
- 平均生存stepは3seedで`82.20`〜`85.05`に収まり、短期生存時間はかなり近い結果でした。
- 前進距離と前進速度はseed間で大きく崩れていませんが、左右移動量、接地切替、歩行候補率には大きな差があります。
- seed `5252`は前進距離は出ていますが、左右移動量が大きく、接地切替が少ないため歩行候補率0%でした。
- seed `6262`は前進距離と左右移動量は良い一方、足滑りが最も大きく、胴体姿勢の平均が最も低いです。
- seed `4242`は歩行候補率が最も高いですが、全エピソードで転倒し、安定成功には届いていません。
- A設定は「短時間だけ前進する傾向」は複数seedで見られますが、「安定した二足歩行」は再現していません。次はAを基準に、胴体高さ低下、接地中の足滑り、制御入力の急変を抑える報酬・カリキュラム改善を検討します。

## フェーズ3.5: action滑らかさ報酬と制御変化解析

フェーズ3.5では、急激な制御指令の変化が転倒や歩行不安定性に関係しているかを調べるため、Stage 1 Aを変更せず、派生設定としてaction差分ペナルティーを追加しました。今回は実装、既存モデルの再評価、1024 stepのsmoke testのみ実施し、100,000 step以上の本格比較学習は開始していません。

追加・変更したファイル:

```text
humanoid_ai\phase3_curriculum_reward.py
configs\phase35_stage1_action_smoothness.json
analyze_phase35_action_smoothness.py
tests\test_phase35_action_smoothness.py
```

報酬設計:

```text
action_delta = action_t - action_{t-1}
初回step: action_{t-1} = 0
episode reset時: action_{t-1}をリセット
action_delta_sq_mean = mean(action_delta^2)
phase3_penalty_action_smoothness = -action_smoothness_weight * action_delta_sq_mean
```

`action`はMuJoCo環境へ渡す実際の制御入力を使い、VecNormalizeの観測正規化とは独立して計測します。既存A設定は`action_smoothness_weight=0.0`のまま扱われ、挙動は変えません。

比較する報酬設定:

| 条件 | 設定ファイル | action_smoothness_weight | 目的 |
| --- | --- | ---: | --- |
| A | `configs\phase31_stage1_A_current.json` | 0.0 | 現行Stage 1基準 |
| Smooth | `configs\phase35_stage1_action_smoothness.json` | 0.2 | actionの急変を弱く抑える |

既存Aモデルのaction差分基準値:

| 学習seed | action差分RMS平均 | p75 | p90 | 足滑り最大平均 | 足滑り-action差分相関 | 関節速度変化RMS | 関節加速度相当RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4242 | 0.384 | 0.466 | 0.505 | 0.138 | -0.038 | 3.178 | 211.897 |
| 5252 | 0.139 | 0.217 | 0.307 | 0.119 | 0.121 | 1.211 | 80.723 |
| 6262 | 0.329 | 0.437 | 0.494 | 0.199 | 0.165 | 3.034 | 202.270 |

重み候補:

- `action_smoothness_weight=0.2`を採用しました。
- この重みでは、p75 action差分に対する追加ペナルティーはseedごとに約`-0.009`〜`-0.043`/step、p90では約`-0.019`〜`-0.051`/stepです。
- Stage 1の速度追従報酬や生存報酬に比べて小さく、強すぎて動作が止まる可能性を避けるための控えめな値です。
- 効果はsmoke testだけでは判断しません。本格比較では、前進距離と生存時間が落ちすぎないかを必ず確認します。

既存3モデルの再評価:

```powershell
.\.venv_phase1\Scripts\python.exe analyze_phase35_action_smoothness.py --episodes 20 --max-steps 1000 --seed 6000 --device cpu --curriculum-stage stage1 --out-dir work\phase35_action_smoothness --graph-dir outputs\phase35_action_smoothness
```

再評価結果:

| 学習seed | 平均生存step | 転倒率 | 歩行候補率 | 安定歩行成功率 | 前進距離 | 左右移動量 | action飽和率 | action差分RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4242 | 83.30 | 100% | 85% | 0% | 0.751 | 0.244 | 50.7% | 0.384 |
| 5252 | 85.05 | 100% | 0% | 0% | 0.778 | 0.410 | 39.4% | 0.139 |
| 6262 | 82.20 | 100% | 70% | 0% | 0.860 | 0.072 | 49.4% | 0.331 |

閾値感度:

| 学習seed | base | strict slip | loose slip | strict lateral | loose lateral | strict upright |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4242 | 85% | 85% | 85% | 50% | 95% | 85% |
| 5252 | 0% | 0% | 0% | 0% | 15% | 0% |
| 6262 | 70% | 25% | 75% | 70% | 70% | 15% |

解釈:

- `walk_candidate`はseed 4242では主に左右移動量の閾値、seed 6262では足滑りと胴体姿勢の閾値に敏感でした。
- seed 5252は横流れが大きく、横方向閾値を緩めても歩行候補率は15%までしか上がりませんでした。
- 3モデルとも安定歩行成功率は0%で、閾値を少し動かしても安定歩行成功とは判定されません。
- 足滑りとaction差分の相関は弱く、4242ではほぼ無相関、5252/6262では弱い正相関でした。action急変だけが不安定性の単独原因とはまだ言えません。

smoke test:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 1024 --seed 4242 --run-name phase35_smooth_smoke --checkpoint-dir work\phase35_smooth_smoke --final-name-prefix phase35_smooth_smoke --log-dir runs\tensorboard --save-freq 512 --device cpu --vec-normalize --curriculum-config configs\phase35_stage1_action_smoothness.json --curriculum-stage stage1

.\.venv_phase1\Scripts\python.exe play.py --agent model --model work\phase35_smooth_smoke\phase35_smooth_smoke_latest.zip --vec-normalize-stats work\phase35_smooth_smoke\phase35_smooth_smoke_vecnormalize.pkl --curriculum-config configs\phase35_stage1_action_smoothness.json --curriculum-stage stage1 --render-mode none --episodes 1 --max-steps 32 --seed 6000 --deterministic --device cpu
```

結果:

| 項目 | 結果 |
| --- | ---: |
| smoke test学習step | 1024 |
| 学習時間 | 4.188秒 |
| steps/sec | 244.519 |
| 保存モデル | `work\phase35_smooth_smoke\phase35_smooth_smoke_latest.zip` |
| VecNormalize統計 | `work\phase35_smooth_smoke\phase35_smooth_smoke_vecnormalize.pkl` |
| 読み込み確認 | 32 step再生成功、terminated=False、truncated=False |

TensorBoardには`phase3_penalty_action_smoothness`、`phase3_action_delta_sq_mean`、`phase3_action_delta_rms`が記録されました。1024 stepは動作確認であり、滑らかさ報酬の改善効果は判断していません。

出力:

```text
work\phase35_action_smoothness\phase35_baseline_summary.csv
work\phase35_action_smoothness\phase35_episode_summary_all.csv
work\phase35_action_smoothness\phase35_action_delta_distribution.csv
work\phase35_action_smoothness\phase35_threshold_sensitivity.csv
work\phase35_action_smoothness\phase35_action_smoothness_analysis.json
outputs\phase35_action_smoothness\phase35_slip_vs_action_delta.svg
```

本格比較実験の推定時間:

| 実験案 | 推定時間 |
| --- | ---: |
| Smooth設定 x 3seed x 100,352 step | 約10〜14分 + 評価1〜2分 |
| AとSmoothを両方新規学習 x 3seed | 約20〜28分 + 評価2〜4分 |

次に進むなら、まずSmooth設定を`4242`, `5252`, `6262`の3seedで100,000 stepずつ学習し、既存Aの3seedと同じ評価指標で比較するのが最小の検証になります。

## フェーズ3.6: action滑らかさペナルティーの3seed対照比較

フェーズ3.6では、Stage 1 A設定と、action差分ペナルティーを加えたSmooth設定を、同じ3つの学習seedで比較しました。A条件はフェーズ3.4の既存モデルを再利用し、Smooth条件のみ新規モデルから学習しました。Stage 2、Stage 3、追加の長時間学習は開始していません。

比較条件:

| 条件 | 設定ファイル | action_smoothness_weight | 学習seed |
| --- | --- | ---: | --- |
| A | `configs\phase31_stage1_A_current.json` | 0.0 | 4242, 5252, 6262 |
| Smooth | `configs\phase35_stage1_action_smoothness.json` | 0.2 | 4242, 5252, 6262 |

共通条件:

```text
MuJoCo Humanoid-v5
PPO + VecNormalize
CPU実行
学習中3D描画なし
各モデル 100,352 step
評価seed 6000〜6019
評価20エピソード/model
deterministic=True
VecNormalize.training=False
VecNormalize.norm_reward=False
最大1000 step
```

速度ベンチマーク:

| 項目 | 結果 |
| --- | ---: |
| ベンチマークstep | 5,120 |
| 実測時間 | 22.231秒 |
| 速度 | 230.310 steps/sec |
| 100,352 step x 3本の推定 | 約21.8分 |

Smooth条件の実測学習時間:

| 学習seed | 学習step | 時間 | steps/sec |
| ---: | ---: | ---: | ---: |
| 4242 | 100,352 | 483.764秒 | 207.440 |
| 5252 | 100,352 | 453.454秒 | 221.306 |
| 6262 | 100,352 | 455.946秒 | 220.096 |

TensorBoardでは、Smooth条件の学習中に`phase3_penalty_action_smoothness`、`phase3_action_delta_sq_mean`、`phase3_action_delta_rms`が記録され、action差分ペナルティーが学習中に実際に計算されていることを確認しました。

評価結果:

| 条件 | seed | 生存step | 転倒率 | 歩行候補率 | 安定歩行成功率 | 前進距離 | 前進速度 | 左右移動量 | 胴体姿勢 | 接地切替 | 足滑り平均 | action差分RMS | action飽和率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 4242 | 83.30 | 100% | 85% | 0% | 0.751 | 0.427 | 0.244 | 0.972 | 4.70 | 0.153 | 0.384 | 50.7% |
| A | 5252 | 85.05 | 100% | 0% | 0% | 0.778 | 0.412 | 0.410 | 0.955 | 1.60 | 0.118 | 0.139 | 39.4% |
| A | 6262 | 82.20 | 100% | 70% | 0% | 0.860 | 0.506 | 0.072 | 0.939 | 3.15 | 0.262 | 0.331 | 49.4% |
| Smooth | 4242 | 66.90 | 100% | 15% | 0% | 0.659 | 0.478 | 0.236 | 0.885 | 2.75 | 0.281 | 0.264 | 40.4% |
| Smooth | 5252 | 75.40 | 100% | 95% | 0% | 0.631 | 0.407 | 0.115 | 0.976 | 3.60 | 0.170 | 0.166 | 48.8% |
| Smooth | 6262 | 68.80 | 100% | 0% | 0% | -0.026 | 0.019 | 0.497 | 0.963 | 1.05 | 0.141 | 0.126 | 36.2% |

3seed平均:

| 条件 | 生存step | 転倒率 | 歩行候補率 | 安定歩行成功率 | 前進距離 | 前進速度 | 左右移動量 | action差分RMS | action飽和率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 83.52 | 100% | 51.7% | 0% | 0.796 | 0.449 | 0.242 | 0.285 | 46.5% |
| Smooth | 70.37 | 100% | 36.7% | 0% | 0.421 | 0.301 | 0.283 | 0.185 | 41.8% |

AからSmoothへの同一seed差分:

| 指標 | 改善seed数/3 | 平均差分 | 解釈 |
| --- | ---: | ---: | --- |
| 生存step | 0 | -13.15 | 全seedで悪化 |
| 転倒率 | 0 | 0.00 | 全seedで100%のまま |
| 安定歩行成功率 | 0 | 0.00 | 全seedで0%のまま |
| 歩行候補率 | 1 | -15.0pt | seed 5252のみ改善 |
| 前進距離 | 0 | -0.375 | 全seedで悪化 |
| 前進速度 | 1 | -0.147 | seed 4242のみ改善 |
| 左右移動量 | 2 | +0.041 | 2seedで改善したが、平均は悪化 |
| 足滑り平均 | 1 | +0.019 | 1seedのみ改善 |
| action差分RMS | 2 | -0.100 | 2seedで滑らかになった |
| action飽和率 | 2 | -4.7pt | 2seedで低下 |

解釈:

- Smooth設定は、目的どおりaction差分RMSとaction飽和率を2/3 seedで下げました。
- ただし、平均生存step、前進距離、前進速度、歩行候補率はAより悪化しました。
- 特にseed 6262では、Smoothによりaction差分と足滑りは下がった一方で、前進距離が`-0.026`となり、前進能力がほぼ失われました。
- seed 5252では歩行候補率と左右移動量、姿勢が改善しましたが、生存stepと前進距離は低下しました。
- 3seedすべてで転倒率100%、安定歩行成功率0%のままであり、安定歩行は未達です。
- actionを滑らかにすること自体は効いていますが、重み`0.2`をそのまま採用して改善とは判定できません。前進能力を失わずに滑らかさを入れるには、より弱い重み、スケジュール型の重み、または胴体高さ・転倒直前姿勢を優先する設計との組み合わせが必要です。

出力:

```text
work\phase36_smoothness_comparison\phase36_model_summary.csv
work\phase36_smoothness_comparison\phase36_episode_summary_all.csv
work\phase36_smoothness_comparison\phase36_same_seed_deltas.csv
work\phase36_smoothness_comparison\phase36_reward_condition_stats.csv
work\phase36_smoothness_comparison\phase36_improvement_counts.csv
work\phase36_smoothness_comparison\phase36_threshold_sensitivity.csv
work\phase36_smoothness_comparison\phase36_smoothness_comparison.json
outputs\phase36_smoothness_comparison\phase36_smooth_minus_A_deltas.svg
outputs\phase36_smoothness_comparison\phase36_A_seed4242_seed6000.gif
outputs\phase36_smoothness_comparison\phase36_A_seed5252_seed6000.gif
outputs\phase36_smoothness_comparison\phase36_A_seed6262_seed6000.gif
outputs\phase36_smoothness_comparison\phase36_Smooth_seed4242_seed6000.gif
outputs\phase36_smoothness_comparison\phase36_Smooth_seed5252_seed6000.gif
outputs\phase36_smoothness_comparison\phase36_Smooth_seed6262_seed6000.gif
```

Smoothモデルの3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase36_smooth_seed4242_100k\phase36_smooth_seed4242_100k_latest.zip --vec-normalize-stats checkpoints\phase36_smooth_seed4242_100k\phase36_smooth_seed4242_100k_vecnormalize.pkl --curriculum-config configs\phase35_stage1_action_smoothness.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase36_smooth_seed5252_100k\phase36_smooth_seed5252_100k_latest.zip --vec-normalize-stats checkpoints\phase36_smooth_seed5252_100k\phase36_smooth_seed5252_100k_vecnormalize.pkl --curriculum-config configs\phase35_stage1_action_smoothness.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005

.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase36_smooth_seed6262_100k\phase36_smooth_seed6262_100k_latest.zip --vec-normalize-stats checkpoints\phase36_smooth_seed6262_100k\phase36_smooth_seed6262_100k_vecnormalize.pkl --curriculum-config configs\phase35_stage1_action_smoothness.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

テスト:

```powershell
.\.venv_phase1\Scripts\python.exe -m py_compile analyze_phase36_smoothness_comparison.py
.\.venv_phase1\Scripts\python.exe tests\test_phase36_smoothness_comparison.py
```

結果:

```text
phase36_smoothness_comparison_tests_done=true
```

## フェーズ3.7: Stage 1 A seed4242の追加100k継続学習

フェーズ3.7では、新しい報酬項を追加せず、Stage 1 A設定のまま追加学習で安定歩行が改善するかを確認しました。対象は`seed=4242`のA 100kモデルのみです。他seed、Smooth条件、Stage 2、Stage 3、200k以降の追加学習は実施していません。

使用した元モデル:

```text
checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_latest.zip
checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_vecnormalize.pkl
configs\phase31_stage1_A_current.json
curriculum-stage=stage1
```

再開確認:

| 項目 | 結果 |
| --- | ---: |
| model_start_num_timesteps | 100,352 |
| optimizer_state_entries | 13 |
| VecNormalize.training | True |
| VecNormalize.norm_reward | True |
| 描画 | 無効 |

独立ベンチマーク:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 5000 --seed 4242 --run-name phase37_A4242_resume_benchmark_5k --checkpoint-dir work\phase37_A4242_resume_benchmark_5k --final-name-prefix phase37_A4242_resume_benchmark_5k --log-dir runs\tensorboard --save-freq 5000 --device cpu --vec-normalize --resume checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1
```

| 項目 | 結果 |
| --- | ---: |
| 追加step | 5,120 |
| 到達step | 105,472 |
| 時間 | 12.480秒 |
| 速度 | 410.255 steps/sec |
| 保存先 | `work\phase37_A4242_resume_benchmark_5k` |

このベンチマークの学習stepは本実験に混入させていません。

本実験:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase25.py --total-timesteps 100000 --seed 4242 --run-name phase37_A4242_resume_200k --checkpoint-dir checkpoints\phase37_A4242_resume_200k --final-name-prefix phase37_A4242_resume_200k --log-dir runs\tensorboard --save-freq 25000 --device cpu --vec-normalize --resume checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase32_A_current_100k\phase32_A_current_100k_seed4242_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1
```

| 項目 | 結果 |
| --- | ---: |
| 開始step | 100,352 |
| 追加step | 100,352 |
| 到達step | 200,704 |
| 実測時間 | 330.652秒 |
| 速度 | 303.497 steps/sec |
| 最終モデル | `checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_latest.zip` |
| 最終VecNormalize統計 | `checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_vecnormalize.pkl` |

中間checkpoint:

```text
checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_ppo_125352_steps.zip
checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_ppo_150352_steps.zip
checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_ppo_175352_steps.zip
checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_ppo_200352_steps.zip
```

各中間モデルに対応するVecNormalize統計も同じフォルダに保存されました。

評価条件:

```text
評価seed: 6000〜6019
20エピソード/checkpoint
deterministic=True
VecNormalize.training=False
VecNormalize.norm_reward=False
最大1000 step
```

評価結果:

| checkpoint | step | 生存step | 転倒率 | 歩行候補率 | 安定歩行成功率 | 前進距離 | 前進速度 | 横流れ | 胴体姿勢 | 接地切替 | 足滑り平均 | action差分RMS | action飽和率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100k | 100,352 | 83.30 | 100% | 85% | 0% | 0.751 | 0.427 | 0.244 | 0.972 | 4.70 | 0.153 | 0.384 | 50.7% |
| A125k | 125,352 | 99.00 | 100% | 75% | 0% | 0.590 | 0.324 | 0.290 | 0.981 | 8.55 | 0.174 | 0.398 | 59.6% |
| A150k | 150,352 | 103.95 | 100% | 80% | 0% | 0.680 | 0.304 | 0.229 | 0.976 | 8.10 | 0.167 | 0.420 | 63.2% |
| A175k | 175,352 | 107.80 | 100% | 70% | 0% | 0.586 | 0.291 | 0.231 | 0.976 | 6.60 | 0.172 | 0.447 | 61.3% |
| A200k_mid | 200,352 | 103.35 | 100% | 60% | 0% | 0.549 | 0.288 | 0.294 | 0.971 | 2.45 | 0.117 | 0.449 | 64.3% |
| A200k | 200,704 | 108.50 | 100% | 65% | 0% | 0.402 | 0.210 | 0.214 | 0.966 | 4.35 | 0.138 | 0.448 | 63.6% |

100kから200kへの変化:

| 指標 | 100k | 200k | 差分 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 生存step | 83.30 | 108.50 | +25.20 | 改善 |
| 転倒率 | 100% | 100% | 0pt | 未改善 |
| 安定歩行成功率 | 0% | 0% | 0pt | 未改善 |
| 歩行候補率 | 85% | 65% | -20pt | 悪化 |
| 前進距離 | 0.751 | 0.402 | -0.348 | 悪化 |
| 前進速度 | 0.427 | 0.210 | -0.218 | 悪化 |
| 横流れ | 0.244 | 0.214 | -0.030 | 改善 |
| 胴体高さ | 1.247 | 1.262 | +0.014 | 改善 |
| 胴体高さ変動 | 0.076 | 0.069 | -0.008 | 改善 |
| 胴体姿勢 | 0.972 | 0.966 | -0.006 | 悪化 |
| 接地切替 | 4.70 | 4.35 | -0.35 | 悪化 |
| 足滑り平均 | 0.153 | 0.138 | -0.015 | 改善 |
| action差分RMS | 0.384 | 0.448 | +0.064 | 悪化 |
| action飽和率 | 50.7% | 63.6% | +12.9pt | 悪化 |

閾値感度:

| checkpoint | base | strict lateral | loose lateral | strict slip | strict upright |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100k | 85% | 50% | 95% | 85% | 85% |
| A125k | 75% | 15% | 90% | 75% | 75% |
| A150k | 80% | 70% | 90% | 80% | 80% |
| A175k | 70% | 45% | 90% | 70% | 70% |
| A200k_mid | 60% | 25% | 90% | 60% | 60% |
| A200k | 65% | 55% | 70% | 65% | 65% |

解釈:

- 追加学習により平均生存stepは`83.30`から`108.50`へ伸びました。
- しかし転倒率は100%のまま、安定歩行成功率も0%のままです。
- 前進距離は`0.751`から`0.402`へ、平均前進速度は`0.427`から`0.210`へ大きく悪化しました。
- 胴体高さ、胴体高さ変動、横流れ、足滑り平均は改善しましたが、action差分RMSとaction飽和率は悪化しました。
- 中間checkpointでは150k付近まで歩行候補率80%を維持していますが、200kでは65%まで低下しました。
- したがって、今回の追加学習は「より長く倒れにくくなる傾向」はありますが、「安定した前進歩行の改善」とは判定できません。以前見られた、生存時間だけ伸びて前進能力が落ちる現象が再発しています。
- 次の改善では、単純な追加学習よりも、前進能力を保つ評価ゲート、action飽和抑制、またはStage 1内での早期停止/ベストcheckpoint選択を優先するべきです。今回の範囲では150k checkpointが200k最終よりバランスが良い候補です。

出力:

```text
work\phase37_longer_baseline\phase37_checkpoint_summary.csv
work\phase37_longer_baseline\phase37_episode_summary_all.csv
work\phase37_longer_baseline\phase37_100k_to_200k_delta.csv
work\phase37_longer_baseline\phase37_threshold_sensitivity.csv
work\phase37_longer_baseline\phase37_longer_baseline.json
outputs\phase37_longer_baseline\phase37_checkpoint_progress.svg
outputs\phase37_longer_baseline\phase37_A100k_seed6000.gif
outputs\phase37_longer_baseline\phase37_A200k_seed6000.gif
```

200kモデルの3D再生コマンド:

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_latest.zip --vec-normalize-stats checkpoints\phase37_A4242_resume_200k\phase37_A4242_resume_200k_vecnormalize.pkl --curriculum-config configs\phase31_stage1_A_current.json --curriculum-stage stage1 --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

テスト:

```powershell
.\.venv_phase1\Scripts\python.exe -m py_compile analyze_phase37_longer_baseline.py
.\.venv_phase1\Scripts\python.exe tests\test_phase37_longer_baseline.py
```

結果:

```text
phase37_longer_baseline_tests_done=true
```

## フェーズ4: 標準Humanoid-v5 PPO基盤

フェーズ4では、報酬関数の細かな調整をいったん停止し、Gymnasium標準の`Humanoid-v5`報酬をそのまま使う再現性重視の学習基盤を追加しました。参考実装として[ProfessorNova/PPO-Humanoid](https://github.com/ProfessorNova/PPO-Humanoid)を確認しましたが、このプロジェクトでは既存方針どおりStable-Baselines3のPPOを使用し、自作PPOには置き換えていません。

追加ファイル:

```text
train_phase4.py
evaluate_phase4.py
tests\test_phase4_cli.py
```

### Phase 4の基本方針

- 環境は`Humanoid-v5`標準報酬のみを使用します。
- 学習は`PPO + VecNormalize`で行います。
- `--total-timesteps`は新規学習時も再開時も「今回追加するstep数」です。
- 再開時は`reset_num_timesteps=False`、`VecNormalize.training=True`、`norm_reward=True`で継続します。
- 評価時は保存済みVecNormalize統計を読み込み、`training=False`、`norm_reward=False`にします。
- 1環境では`DummyVecEnv`、2環境以上ではWindows対応の`SubprocVecEnv(spawn)`を使います。
- checkpointごとにPPOモデル、VecNormalize統計、メタデータJSONをセットで保存します。
- Ctrl+C中断時もinterrupt checkpointを保存します。

### プリセット

`baseline`:

```text
n_steps=512
batch_size=64
n_epochs=5
learning_rate=3e-4
net_arch=[64,64]
gamma=0.99
gae_lambda=0.95
clip_range=0.2
vf_coef=0.5
max_grad_norm=0.5
```

`candidate`:

```text
n_steps=1024
batch_size=256
n_epochs=10
learning_rate=1e-4
ent_coef=0.01
net_arch=[256,256]
gamma=0.99
gae_lambda=0.95
clip_range=0.2
vf_coef=0.5
max_grad_norm=0.5
```

これらは比較用の初期設定であり、最適値ではありません。初回比較では条件を単純にするため固定学習率を使います。

### A. baselineで新規100k学習

```powershell
.\.venv_phase1\Scripts\python.exe train_phase4.py --preset baseline --total-timesteps 100000 --seed 4242 --n-envs 4 --device cpu --checkpoint-dir checkpoints\phase4_baseline_100k_seed4242 --run-name phase4_baseline_100k_seed4242 --save-freq 25000 --torch-threads 10
```

### B. candidateで新規100k学習

```powershell
.\.venv_phase1\Scripts\python.exe train_phase4.py --preset candidate --total-timesteps 100000 --seed 4242 --n-envs 4 --device cpu --checkpoint-dir checkpoints\phase4_candidate_100k_seed4242 --run-name phase4_candidate_100k_seed4242 --save-freq 25000 --torch-threads 10
```

### C. 既存checkpointから追加200k学習

```powershell
.\.venv_phase1\Scripts\python.exe train_phase4.py --preset baseline --total-timesteps 200000 --seed 4242 --n-envs 4 --device cpu --checkpoint-dir checkpoints\phase4_baseline_resume_300k_seed4242 --run-name phase4_baseline_resume_300k_seed4242 --resume checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_latest_vecnormalize.pkl --save-freq 25000 --torch-threads 10
```

### D. 保存モデルの20エピソード評価

```powershell
.\.venv_phase1\Scripts\python.exe evaluate_phase4.py --model checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_latest_vecnormalize.pkl --episodes 20 --seed 6000 --device cpu --out-dir work\phase4_eval_baseline_100k --run-name phase4_eval_baseline_100k
```

評価出力:

```text
phase4_step_log.csv
phase4_episode_summary.csv
phase4_contact_timeline.svg
phase4_summary.json
```

### E. 3D再生

```powershell
.\.venv_phase1\Scripts\python.exe play.py --agent model --model checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_latest.zip --vec-normalize-stats checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_latest_vecnormalize.pkl --render-mode human --episodes 1 --max-steps 1000 --seed 6000 --deterministic --device cpu --sleep 0.005
```

### F. TensorBoard確認

```powershell
.\.venv_phase1\Scripts\tensorboard.exe --logdir runs\tensorboard
```

ブラウザで表示:

```text
http://localhost:6006
```

### G. Ctrl+C中断と再開

学習中にCtrl+Cを押すと、次のようなinterrupt checkpointを保存して終了します。

```text
<checkpoint-dir>\<run-name>_interrupt.zip
<checkpoint-dir>\<run-name>_interrupt_vecnormalize.pkl
<checkpoint-dir>\<run-name>_interrupt_metadata.json
```

再開例:

```powershell
.\.venv_phase1\Scripts\python.exe train_phase4.py --preset baseline --total-timesteps 100000 --seed 4242 --n-envs 4 --device cpu --checkpoint-dir checkpoints\phase4_baseline_resume_after_interrupt --run-name phase4_baseline_resume_after_interrupt --resume checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_interrupt.zip --vec-normalize-stats checkpoints\phase4_baseline_100k_seed4242\phase4_baseline_100k_seed4242_interrupt_vecnormalize.pkl --save-freq 25000 --torch-threads 10
```

### 短い動作確認

実施した確認:

```powershell
.\.venv_phase1\Scripts\python.exe -m py_compile train_phase4.py evaluate_phase4.py
.\.venv_phase1\Scripts\python.exe train_phase4.py --help
.\.venv_phase1\Scripts\python.exe evaluate_phase4.py --help
.\.venv_phase1\Scripts\python.exe tests\test_phase4_cli.py
```

結果:

```text
phase4_cli_tests_done=true
```

smoke test:

| 条件 | step | 実測時間 | 速度 |
| --- | ---: | ---: | ---: |
| baseline, n_envs=1 | 1,024 | 2.422秒 | 422.751 steps/sec |
| candidate, n_envs=1 | 1,024 | 2.644秒 | 387.248 steps/sec |
| baseline resume | 1,024追加 | 2.014秒 | 508.350 steps/sec |

保存・再開確認:

- baseline smokeは`work\phase4_smoke_baseline`へモデル、VecNormalize統計、metadataを保存しました。
- resume smokeは`model_start_num_timesteps=1024`から開始し、`trained_timesteps=2048`で終了しました。
- resume時の`optimizer_state_entries=13`を確認し、PPO optimizer状態が保存モデルから読み込まれていることを確認しました。
- 評価時は`VecNormalize.training=False`、`VecNormalize.norm_reward=False`で実行できました。
- `play.py --render-mode none`でPhase 4保存モデルとVecNormalize統計の読み込みを確認しました。

### n_envsベンチマーク

`baseline`で約5,000stepの短い速度計測を行いました。PPO rollout単位により実step数は指定値より少し増える場合があります。

| n_envs | VecEnv | 実step | 実測時間 | 速度 |
| ---: | --- | ---: | ---: | ---: |
| 1 | DummyVecEnv | 5,120 | 10.900秒 | 469.732 steps/sec |
| 2 | SubprocVecEnv | 5,120 | 12.323秒 | 415.482 steps/sec |
| 4 | SubprocVecEnv | 6,144 | 9.503秒 | 646.518 steps/sec |

短い計測では`n_envs=4`が最速でした。ただし長時間学習ではCPU温度やメモリ、プロセス間通信の影響で速度が変わる可能性があります。発熱や操作の重さが出る場合は`n_envs=1`または`2`に下げてください。

`n_envs=4`、646.518 steps/secを基準にした概算:

| step | 推定時間 |
| ---: | ---: |
| 100,000 | 約2.6分 |
| 300,000 | 約7.7分 |
| 1,000,000 | 約25.8分 |

未確認・注意:

- 今回は100,000step以上の本格学習は実行していません。
- 3Dウィンドウ表示はPhase 4では未実行です。モデル読み込みは`render-mode none`で確認済みです。
- WindowsのCIM権限の都合でRAM容量の自動取得は失敗しました。長時間学習時はタスクマネージャーでCPU温度、使用率、メモリを確認してください。
- Python 3.14環境では`Failed to find real location...`警告が出ますが、今回のテストでは処理は正常終了しました。

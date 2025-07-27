# NovelAI Batch Image Generator

NovelAI APIを使用して複数の画像を一括生成するPythonスクリプトです。

## セットアップ

1. 必要なライブラリをインストール:
```bash
pip install -r requirements.txt
```

2. `.env.example`を`.env`にコピーして、NovelAI APIトークンを設定:
```bash
cp .env.example .env
# .envファイルを編集してNOVELAI_TOKENを設定
```

## 使い方

### 基本的な使用方法（100枚生成）
```bash
python novelai_batch_generator.py
```

### カスタム設定での使用
```bash
python novelai_batch_generator.py \
  --count 50 \
  --model NAI_DIFFUSION_4_5_CURATED \
  --width 1024 \
  --height 1536 \
  --steps 23 \
  --scale 5.0
```

### V4.5 Curatedモデルを使用する場合
```bash
python novelai_batch_generator.py \
  --count 100 \
  --model NAI_DIFFUSION_4_5_CURATED
```

### コマンドラインオプション
- `--prompt-file`: プロンプトファイルのパス（デフォルト: prompts.txt）
- `--negative-prompt-file`: ネガティブプロンプトファイルのパス（デフォルト: negative_prompts.txt）
- `--count`: 生成する画像数（デフォルト: 100）
- `--model`: 使用するモデル（デフォルト: NAI_DIFFUSION_4_5_FULL）
- `--width`: 画像の幅（デフォルト: 832）
- `--height`: 画像の高さ（デフォルト: 1216）
- `--steps`: サンプリングステップ数（デフォルト: 23）
- `--scale`: ガイダンススケール（デフォルト: 5.0）
- `--sampler`: サンプラー（デフォルト: k_euler_ancestral）
- `--output-dir`: カスタム出力ディレクトリ
- `--trials`: プロンプトごとの試行回数（異なるシードで生成）
- `--batch-name`: バッチ名（ファイル名に使用）

## 利用可能なモデル
- NAI_DIFFUSION_4_5_FULL（デフォルト）- NovelAI Diffusion V4.5 Full（最新・高品質）
- NAI_DIFFUSION_4_5_CURATED - NovelAI Diffusion V4.5 Curated（最新・軽量版）

## ファイル構成
- `prompts.txt`: プロンプト（1行1プロンプト）
- `negative_prompts.txt`: ネガティブプロンプト（1行1プロンプト）
- `config.json`: デフォルト設定
- `output/`: 生成された画像の出力先
  - `batch_YYYYMMDD_HHMMSS/`: タイムスタンプ付きフォルダ
    - `images/`: 生成画像
    - `generation_config.json`: 実行時の設定
    - `prompts_used.txt`: 使用したプロンプト
    - `negative_prompts_used.txt`: 使用したネガティブプロンプト
    - `generation_log.txt`: 実行ログ

## 注意事項
- NovelAI APIの利用にはアカウントとAPIトークンが必要です
- 大量の画像を生成する場合は、APIの利用制限に注意してください
- NovelAI APIの仕様により同時実行はできませんが、非同期処理により順次効率的に生成します
- デフォルトモデルはV4.5 Fullに設定されています。軽量版を使用したい場合は`--model NAI_DIFFUSION_4_5_CURATED`を指定するか、`config.json`の`"model"`を変更してください

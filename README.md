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
  --model NAI_DIFFUSION_3 \
  --width 768 \
  --height 1024 \
  --steps 28
```

### V4.5モデルを使用する場合
```bash
python novelai_batch_generator.py \
  --count 100 \
  --model NAI_DIFFUSION_4_5_CURATED_PREVIEW
```

### コマンドラインオプション
- `--prompt-file`: プロンプトファイルのパス（デフォルト: prompts.txt）
- `--negative-prompt-file`: ネガティブプロンプトファイルのパス（デフォルト: negative_prompts.txt）
- `--count`: 生成する画像数（デフォルト: 100）
- `--model`: 使用するモデル（デフォルト: NAI_DIFFUSION_4_5_FULL）
- `--width`: 画像の幅（デフォルト: 512）
- `--height`: 画像の高さ（デフォルト: 768）
- `--steps`: サンプリングステップ数（デフォルト: 28）
- `--scale`: ガイダンススケール（デフォルト: 11.0）
- `--sampler`: サンプラー（デフォルト: k_euler_ancestral）
- `--output-dir`: カスタム出力ディレクトリ

## 利用可能なモデル
- NAI_DIFFUSION_3（デフォルト）- NovelAI Diffusion V3
- NAI_DIFFUSION_4_5_CURATED_PREVIEW - NovelAI Diffusion V4.5 Curated Preview（最新）
- NAI_DIFFUSION_4_CURATED_PREVIEW - NovelAI Diffusion V4 Curated Preview
- NAI_DIFFUSION_3_INPAINTING - NovelAI Diffusion V3 Inpainting
- NAI_DIFFUSION - NovelAI Diffusion V1
- NAI_DIFFUSION_2 - NovelAI Diffusion V2
- SAFE_DIFFUSION - Safe Diffusion (Curated)
- NAI_DIFFUSION_FURRY - NovelAI Diffusion Furry

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
- 非同期処理により効率的に生成しますが、同時接続数は5に制限されています
- V4.5モデルをデフォルトにしたい場合は、`config.json`の`"model"`を`"NAI_DIFFUSION_4_5_CURATED_PREVIEW"`に変更してください
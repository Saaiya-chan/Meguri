# Meguri — 静的サイト（Astro）

議論ドキュメントを Markdown（フロントマター付き）からビルドする静的サイトです。

## 使い方（ルート `index.html` から閲覧）

プロジェクトルートの `index.html` から、各カードの「議論を読む」「討論を読む」「レビューを読む」「対応策を確認」で、Astro でビルドした Markdown 版（`site/dist/`）に遷移できます。

## コンテンツ

- **議論:** `src/content/discussions/` に Markdown を配置。フロントマターは `title`, `date`, `description`, `tags`。
- ルートの `discussions/*.md` を変換したら、このディレクトリにコピーするか、シンボリックリンクで参照してください。

## HTML を追加した場合（HTML → Markdown → 一覧へ反映）

※以下のコマンドはプロジェクトルート（`Meguri_pre3/`）で実行する想定です。

1. `discussions/新規.html` を追加
2. Markdown 化: `python3 scripts/html_to_md.py discussions/新規.html`（既定で `discussions/新規.md` を生成）
3. 元 HTML を退避: `mv discussions/新規.html discussions/OLD/`
4. コンテンツへ反映: `cp discussions/新規.md site/src/content/discussions/`（またはシンボリックリンク）
5. ビルド: `cd site && npm run build`（`site/dist/` と一覧が更新されます）

## コマンド

| コマンド | 説明 |
|----------|------|
| `npm run dev` | 開発サーバー（localhost:4321） |
| `npm run build` | 静的ファイルを `dist/` に出力 |
| `npm run preview` | ビルドのプレビュー |

## 数式

KaTeX を使用（`remark-math` + `rehype-katex`）。Markdown 内の `$$...$$` がレンダリングされます。

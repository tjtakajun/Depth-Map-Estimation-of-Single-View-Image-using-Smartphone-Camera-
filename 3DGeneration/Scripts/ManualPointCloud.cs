using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class ManualPointCloud : MonoBehaviour
{
    public Texture2D depthImage; // 手動で準備した白黒画像をInspectorからアタッチする
    public Material pointCloudMaterial; // ポイントクラウドのマテリアルをInspectorからアタッチする

    private Mesh _mesh;
    public RawImage depth;
    private GameObject DepthMap;
    private List<Vector3> _vertices = new List<Vector3>();
    private List<Color32> _colors = new List<Color32>();
    private List<int> _indices = new List<int>();

   // private void Start()
   //{
   //     DepthMap = GameObject.Find("DepthMap");
   // }

    public GameObject InitializePointCloud()
    {
        // メッシュを作成
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        // 画像をテクスチャとしてマテリアルに設定
        pointCloudMaterial.SetTexture("_DepthTex", depthImage);

        // ポイントクラウドの生成
        CalculatePointCloud();

        // 点群のメッシュを作成してAR空間に配置
        GameObject pointCloudObject = new GameObject("PointCloud");
        MeshFilter meshFilter = pointCloudObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = pointCloudObject.AddComponent<MeshRenderer>();
        meshFilter.mesh = _mesh;
        meshRenderer.material = pointCloudMaterial;

        // ポイントクラウドオブジェクトの位置をAR空間に合わせる
        pointCloudObject.transform.position = transform.position;
        pointCloudObject.transform.rotation = transform.rotation;

        return pointCloudObject; // 生成した点群オブジェクトを返す
    }

    public void PickDepthMap()
    {
        // 画像の読み込み
        NativeGallery.Permission permission = NativeGallery.GetImageFromGallery((path) =>
        {
            UnityEngine.Debug.Log("Image path: " + path);
            if (path != null)
            {
                // 画像パスからTexture2Dを生成
                Texture2D texture = NativeGallery.LoadImageAtPath(path, 512);
                if (texture == null)
                {
                    UnityEngine.Debug.Log("Couldn't load texture from " + path);
                    return;
                }

                // RawImageの元テクスチャを破棄
                Destroy(depth.texture);

                // RawImageで新規テクスチャを表示
                depth.texture = texture;
                depthImage = createReadabeTexture2D(texture);
            }
        });
        UnityEngine.Debug.Log("Permission result: " + permission);
    }

    Texture2D createReadabeTexture2D(Texture2D texture2d)
    {
        RenderTexture renderTexture = RenderTexture.GetTemporary(
                    texture2d.width,
                    texture2d.height,
                    0,
                    RenderTextureFormat.Default,
                    RenderTextureReadWrite.Linear);

        Graphics.Blit(texture2d, renderTexture);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = renderTexture;
        Texture2D readableTextur2D = new Texture2D(texture2d.width, texture2d.height);
        readableTextur2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        readableTextur2D.Apply();
        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(renderTexture);
        return readableTextur2D;
    }



    void CalculatePointCloud()
    {
        Color32[] pixels = depthImage.GetPixels32();

        // 画像の幅と高さを取得
        int width = depthImage.width;
        int height = depthImage.height;

        // 新しい点群の幅と高さを設定
        int newWidth = width * 2;
        int newHeight = height * 2;

        // 輝度値の最小値と最大値を取得
        float minBrightness = float.MaxValue;
        float maxBrightness = float.MinValue;

        // 輝度値の最小値と最大値を検索
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                Color32 pixel = pixels[index];

                // RGB値の平均を輝度値として取得
                float brightness = (pixel.r + pixel.g + pixel.b) / 3f;

                minBrightness = Mathf.Min(minBrightness, brightness);
                maxBrightness = Mathf.Max(maxBrightness, brightness);
            }
        }

        // ポイントクラウドの生成
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                // 補間対象の画素座標を計算
                int srcX = Mathf.FloorToInt(x / 2f);
                int srcY = Mathf.FloorToInt(y / 2f);

                // 輝度値を補間して取得
                float normalizedBrightness = InterpolateBrightness(pixels, srcX, srcY, width, height, minBrightness, maxBrightness);

                // 3D空間の座標を計算（簡単にx軸とy軸はピクセル座標、z軸は正規化された輝度値とする）
                Vector3 vertex = new Vector3(x, y, normalizedBrightness);

                // ポイントクラウドの頂点リストと色情報リストに追加
                _vertices.Add(vertex);
                _colors.Add(pixels[srcY * width + srcX]);

                // インデックス情報を追加
                _indices.Add(_vertices.Count - 1);
            }
        }

        Debug.Log("Vertex count: " + _vertices.Count);
        for (int i = 0; i < _vertices.Count; i++)
        {
            Debug.Log("Vertex " + i + ": " + _vertices[i] + ", Color: " + _colors[i]);
        }

        // メッシュの頂点情報と色情報を設定
        _mesh.SetVertices(_vertices);
        _mesh.SetColors(_colors);
        _mesh.SetIndices(_indices.ToArray(), MeshTopology.Points, 0);
    }

    // 輝度値を補間して取得するメソッド
    private float InterpolateBrightness(Color32[] pixels, int x, int y, int width, int height, float minBrightness, float maxBrightness)
    {
        // 画素座標が画像範囲外の場合、補間しないで輝度値の最小値を返す
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return minBrightness;
        }

        int index = y * width + x;
        Color32 pixel = pixels[index];

        // RGB値の平均を輝度値として取得
        float brightness = (pixel.r + pixel.g + pixel.b) / 3f;

        // 輝度値を0から1に正規化
        float normalizedBrightness = Mathf.InverseLerp(minBrightness, maxBrightness, brightness);

        // 厚みの倍率を設定（1より大きい値を設定すると厚みが大きくなります）
        float thicknessMultiplier = 21f;

        // 正規化された輝度値に倍率をかける
        normalizedBrightness *= thicknessMultiplier;

        return normalizedBrightness;
    }
}

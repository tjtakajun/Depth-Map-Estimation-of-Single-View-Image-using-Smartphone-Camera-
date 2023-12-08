using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ThreeDMeshGen : MonoBehaviour
{
    public GameObject DepthCursor;
    public GameObject PointCloud;
    public Material PointCloudMaterial;
    public GameObject Waypoint;

    private const float _avatarOffset = 0.9f;
    private const float _waypointYOffset = 0.05f;
    private GameObject _root;
    private bool _firstWaypointPlaced;

    public void DropWaypoint()
    {
        if (_root == null)
        {
            _root = new GameObject("Waypoints");
        }

        Vector3 pos = DepthCursor.transform.position;
        pos.y += _waypointYOffset;

        // PointCloudオブジェクトを生成
        GameObject pointCloudObject = Instantiate(PointCloud, pos, Quaternion.identity);

        // DepthCursorの位置から少し手前に移動させる
        Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
        pointCloudObject.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);

        // PointCloudオブジェクトを有効にする
        pointCloudObject.SetActive(true);

        _firstWaypointPlaced = true;

        // ポイントクラウドが生成されたかどうかのログを出力
        if (pointCloudObject != null)
        {
            Debug.Log("PointCloud object created successfully at DepthCursor position.");
        }
        else
        {
            Debug.LogError("Failed to create PointCloud object at DepthCursor position.");
        }
    }


    public void Clear()
    {
        if (_root != null)
        {
            foreach (Transform child in _root.transform)
            {
                Destroy(child.gameObject);
            }
            _firstWaypointPlaced = false; // ポイントクラウドをクリアしたので、_firstWaypointPlacedをリセットする
        }
    }

    private void OnDestroy()
    {
        Destroy(_root);
        _root = null;
    }

    private void Start()
    {
        // ManualPointCloud スクリプトを持つオブジェクトを取得
        
        //if (_manualPointCloud == null)
        //{
         //   Debug.LogError("ManualPointCloud script not found in the scene.");
         //   return;
        //}

       // PointCloud = _manualPointCloud.InitializePointCloud(); // 初期化メソッドを呼び出す

        // ManualPointCloud が生成した点群オブジェクトにマテリアルを設定
        MeshRenderer meshRenderer = PointCloud.GetComponent<MeshRenderer>();
        meshRenderer.material = PointCloudMaterial;
        // 点群が生成されたかどうかのログを出力
        if (PointCloud != null)
        {
            Debug.Log("PointCloud object created successfully.");
        }
        else
        {
            Debug.LogError("Failed to create PointCloud object.");
        }
        // ポイントクラウドオブジェクトのスケールを調整
        float pointCloudScale = 0.030f; // 例として 0.001 をスケールとして使用
        PointCloud.transform.localScale = new Vector3(pointCloudScale, pointCloudScale, pointCloudScale);

    }


    public void NewPointCloud()
    {
        var start_time = DateTime.Now;
        // ManualPointCloud スクリプトを持つオブジェクトを取得
        //_manualPointCloud = FindObjectOfType<ManualPointCloud>();
        //PointCloud = _manualPointCloud.InitializePointCloud(); // 画像変更メソッドを呼び出す

        // ManualPointCloud が生成した点群オブジェクトにマテリアルを設定
        MeshRenderer meshRenderer = PointCloud.GetComponent<MeshRenderer>();
        meshRenderer.material = PointCloudMaterial;
        // 点群が生成されたかどうかのログを出力
        if (PointCloud != null)
        {
            Debug.Log("Mesh object created successfully 2.");
        }
        else
        {
            Debug.LogError("Failed to create Mesh object 2.");
        }
        // メッシュオブジェクトのスケールを調整
        float pointCloudScale = 0.030f; // 例として 0.001 をスケールとして使用
        PointCloud.transform.localScale = new Vector3(pointCloudScale, pointCloudScale, pointCloudScale);
        _firstWaypointPlaced = false;
        UnityEngine.Debug.Log(String.Format("Took {0} seconds to complete", DateTime.Now - start_time));
    }

    private void Update()
    {
        if (!_firstWaypointPlaced)
        {
            Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
            PointCloud.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);
            // 90度回転するためのクォータニオンを作成
            //Quaternion rotation = Quaternion.Euler(0f, 90f, 0f);

            // PointCloudオブジェクトを回転させる
            // PointCloud.transform.rotation *= rotation;
        }
    }
}

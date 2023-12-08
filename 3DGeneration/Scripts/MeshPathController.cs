using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshPathController : MonoBehaviour
{
    public GameObject DepthCursor;
    public GameObject MeshObject; // MeshObjectに深度画像から生成したメッシュを設定する
    public Material MeshMaterial;
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

        // MeshObjectを生成
        GameObject meshObject = Instantiate(MeshObject, pos, Quaternion.identity);

        // DepthCursorの位置から少し手前に移動させる
        Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
        meshObject.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);

        // MeshObjectを有効にする
        meshObject.SetActive(true);

        _firstWaypointPlaced = true;
    }

    public void Clear()
    {
        if (_root != null)
        {
            foreach (Transform child in _root.transform)
            {
                Destroy(child.gameObject);
            }
        }
    }

    private void OnDestroy()
    {
        Destroy(_root);
        _root = null;
    }

    private void Start()
    {
        // DepthMapToMesh スクリプトを持つオブジェクトを取得
        DepthMapToMesh depthMapToMesh = FindObjectOfType<DepthMapToMesh>();
        if (depthMapToMesh == null)
        {
            Debug.LogError("DepthMapToMesh script not found in the scene.");
            return;
        }

        // DepthMapToMeshで生成したメッシュをMeshObjectに設定
        MeshObject = depthMapToMesh.gameObject;

        // MeshObject が生成したメッシュオブジェクトにマテリアルを設定
        MeshRenderer meshRenderer = MeshObject.GetComponent<MeshRenderer>();
        meshRenderer.material = MeshMaterial;
    }

    private void Update()
    {
        if (!_firstWaypointPlaced)
        {
            Vector3 toCamera = DepthSource.ARCamera.transform.position - DepthCursor.transform.position;
            MeshObject.transform.position = DepthCursor.transform.position + (toCamera.normalized * _avatarOffset);
        }
    }
}

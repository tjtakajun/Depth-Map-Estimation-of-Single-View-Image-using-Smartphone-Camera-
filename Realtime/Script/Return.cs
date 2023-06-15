using System;
using System.Collections;
using System.Collections.Generic;
using Google.XR.ARCoreExtensions;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;


public class Return : MonoBehaviour
{
    private GameObject CarouselUI;

    public void ReturnToCapture()
    {
        //SceneManager.LoadScene("DemoCarousel");
        CarouselUI.SetActive(true);
        // ARSceneÇ©ÇÁARSessionOriginÇéÊìæÇµÅCtrueÇ…Ç∑ÇÈ
        GameObject sessionOriginObject = GameObject.Find("ARCamera");
        ARSessionOrigin sessionOrigin = sessionOriginObject.GetComponent<ARSessionOrigin>();
        sessionOrigin.enabled = true;
    }
    // Start is called before the first frame update
    void Start()
    {
        CarouselUI = GameObject.Find("Carousel UI");
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

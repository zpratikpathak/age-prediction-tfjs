let model;
function openCvReady() {
  cv['onRuntimeInitialized']= ()=>{
    let video = document.getElementById("cam_input"); // video is the id of video tag
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occurred! " + err);
    });
    let modelURL='https://teachablemachine.withgoogle.com/models/9NRc5H8KQ/';
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(cam_input);
    let faces = new cv.RectVector();
    let predictions="Detecting..."
    let f=["Aman","veer","ayush"]
    let arr=[]
    let sarr=[]
    let classifier = new cv.CascadeClassifier();
    let utils = new Utils('errorMessage');
    let crop=new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let dsize = new cv.Size(224, 224);
    
    let faceCascadeFile = 'haarcascade_frontalface_default.xml'; // path to xml
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
    classifier.load(faceCascadeFile); // in the callback, load the cascade from file 
});

// async function loadModel(){
//     let promise = new Promise((resolve,reject)=>{
//         model= ml5.imageClassifier(modelURL+'model.json')
//     })
    
//     model=await promise
//     console.log(model)
//     alert(result)

//     }
// loadModel();
(async () => {
   model = await ml5.imageClassifier(modelURL+'model.json')//,video=video)
   console.log(model)
 })()

 async function predict(img){
     predictions=await model.classify(img)
     return predic
 }

 function sortA(arr){
     let min=arr[0];
     let max=arr[0];
     for (let i=0;i<3;i++){
         if(min>arr[i]){
             min=arr[i]
         }
         else if(max<arr[i]){
             max=arr[i]
         }
     }
     return [min,max]
 }
    const FPS = 30;
    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        try{
            classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
           // console.log(faces.size());
        
        }catch(err){
            console.log(err);
        }
        if(faces.size()==3){
            arr=[faces.get(0).x,faces.get(1).x,faces.get(2).x]
        sarr=sortA(arr)
        console.log(sarr)
        }
        if(faces.size()==2){
            arr=[faces.get(0).x,faces.get(1).x]
            sarr=sortA(arr)
        }
        
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            if(face.width*face.height < 40000){
                continue;}
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(dst, point1, point2, [51, 255, 255, 255],3);
            let cutrect=new cv.Rect(face.x,face.y,face.width,face.height)
    
            crop=dst.roi(cutrect)
           // cv.resize(crop, crop, dsize, 0, 0, cv.INTER_AREA);
            c_temp=document.createElement('canvas')
            c_temp.setAttribute("width",224)
            c_temp.setAttribute("height",224)
            ctx_temp=c_temp.getContext('2d')
            
            if(faces.size()==3){

            if(face.x==sarr[0]){
                cv.putText(dst,String(f[1]).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
            }
            else if(face.x==sarr[1]){
                cv.putText(dst,String(f[2]).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
            }
            else{
                cv.putText(dst,String(f[0]).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
            }
        }
        else if(faces.size()==2){
            if(face.x==sarr[0]){
                cv.putText(dst,String(f[1]).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
            }
            else{
                cv.putText(dst,String(f[0]).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
            }
        }
        else{
            cv.putText(dst,String(f[i]).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);
        }
            let imgData = new ImageData(new Uint8ClampedArray(crop.data), crop.cols, crop.rows);
            ctx_temp.putImageData(imgData,0,0)
            // let canvasToImage=new Image();
            // canvasToImage.src = ctx_temp.toDataURL();

            
            // if(model){
            // predict(ctx_temp)
            // console.log("Detected face ",i)
            // console.log(ctx_temp)
            // console.log(predictions)
            // cv.putText(dst,String(predictions[0].label).toUpperCase(),{x:face.x,y:face.y-20},1,3,[255, 128, 0, 255],4);}
           
        }
      


        cv.imshow("canvas_output", dst);
       
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
}
// schedule first one.
setTimeout(processVideo, 0);
  };
}
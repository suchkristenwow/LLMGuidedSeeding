document.addEventListener("DOMContentLoaded", function() {
    const videoElement = document.getElementById('VideoN');
    const pauseButton = document.getElementById('pause-button');
    const pauseImg = document.getElementById('pause-img');
    let isPaused = false;

    // Retrieve URLs from data attributes
    const initialImage = pauseButton.getAttribute('data-initial-image');
    const pausedImage = pauseButton.getAttribute('data-paused-image');

    // console.log(pausedImage)
    // console.log(pauseButton)

    pauseButton.addEventListener('click', function(e) {
        e.preventDefault();
        isPaused = !isPaused;
        if (isPaused) {
            fetch("http://127.0.0.1:5000/backend/pause").then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }}).catch(error => {
            console.error('There was a problem with the fetch operation:', error);});

            videoElement.src = "/backend/pause";
            pauseImg.src = pausedImage;
        } else {
            videoElement.src = "/backend/image_stream";
            pauseImg.src = initialImage;
        }
    });
});

// function togglePausePlay() {
//     const imgElement = document.getElementById('pause-image');
//     if (imgElement.src.includes('pause.png')) {
//         imgElement.src = "{{ url_for('static', filename='images/play.png') }}";
//     } else {
//         imgElement.src = "{{ url_for('static', filename='images/pause.png') }}";
//     }
// };



// document.addEventListener("DOMContentLoaded", function() {
//     const videoElement = document.getElementById('VideoN');
//     const pauseButton = document.getElementById('pause-button');
//     let isPaused = false;

//     // Initial state of the button
//     const initialImage = "{{ url_for('static', filename='images/pause.png') }}";
//     const pausedImage = "{{ url_for('static', filename='images/play.png') }}";
//     pauseButton.querySelector('img').src = initialImage;

//     // Click event listener for pause/play functionality
//     pauseButton.addEventListener('click', function(e) {
//         e.preventDefault(); // Prevent default anchor behavior

//         // Toggle pause/play state
//         isPaused = !isPaused;
//         if (isPaused) {
//             videoElement.src = "/backend/pause";
//             pauseButton.querySelector('img').src = pausedImage; // Change to play button image
//         } else {
//             videoElement.src = "/backend/image_stream";
//             pauseButton.querySelector('img').src = initialImage; // Change back to pause button image
//         }
//     });
// });

// document.addEventListener("DOMContentLoaded", function() {
    //     const videoElement = document.getElementById('VideoN');
    //     const pauseButton = document.getElementById('pause-button');
    //     let isPaused = false;
    
    //     // // Initial state of the button
    //     // const initialImage = "{{ url_for('static', filename='../images/pause.png') }}";
    //     // const pausedImage = "{{ url_for('static', filename='../images/play.png') }}";
    //     pauseButton.querySelector('img').src = initialImage;
    
    //     pauseButton.addEventListener('click', function(e) {
    //         e.preventDefault();
    //         isPaused = !isPaused;
    //         if (isPaused) {
    //             videoElement.src = "/backend/pause";
    //             // pauseButton.querySelector('img').src = pausedImage; // Change to play button image
    //         } else {
    //             videoElement.src = "/backend/image_stream";
    //             // pauseButton.querySelector('img').src = initialImage; // Change back to pause button image
    //         }
    //     });
    // });

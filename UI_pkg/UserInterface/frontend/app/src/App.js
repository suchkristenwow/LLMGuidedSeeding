// useState: hook to manage the state of the img source
// useEffect: hook to fetch the image when the component mounts
import React, { useState, useRef, useReducer, useEffect} from 'react';
// import axios from 'axios';
import './App.css';
import io from 'socket.io-client'
// https://www.youtube.com/watch?v=Bv8FORu-ACA
// https://www.youtube.com/watch?v=me-BX6FtA9o

function ImageStream() {
  const [paused, setPaused] = useState(false)
  const [drawing, setDrawing] = useState(false) 
  const [url, setUrl] = useState("http://127.0.0.1:5000/backend/image_stream");

  const togglePause = () => {
    setPaused(paused => !paused); // Toggle the paused state
    
    // Construct the URL based on the new paused state
    const newUrl = paused ? "http://127.0.0.1:5000/backend/image_stream" : "http://127.0.0.1:5000/backend/pause";
    setUrl(newUrl);
    
    // If the new state is not paused, trigger a fetch request to pause
    if (!paused) {
      fetch(newUrl)
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
        })
        .catch(error => {
          console.error('There was a problem with the fetch operation:', error);
        });
    } else {
      setDrawing(drawing => false)
    }
    }

  const toggleDrawing = () => {
    // if not in drawing mode switch to drawing mode
    const newUrl = (paused && drawing) ? "http://127.0.0.1:5000/backend/pause":"http://127.0.0.1:5000/backend/sketch_boundary"; 
    setUrl(newUrl)
    setDrawing(drawing => !drawing)
    if (drawing){
      fetch("http://127.0.0.1:5000/backend/sketch_boundary")
      .then(response => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
      })
      .catch(error => {
        console.error("There was a problem with the fetch operation:", error);
      });
        }
    }
  
  
  const paused_key = paused ? "Paused" : "Playing"; // Unique key for image component
  const drawing_key = drawing ? "Drawing": "Not Drawing"
 
  return(
    <div className="imageStream">
      <p>{paused_key}</p>
      <p>{drawing_key}</p>
  <img className = "display_class" 
    id="VideoN" 
    src = {url} 
    alt = "Video Stream" >
  </img>
  <button onClick={togglePause}>
  {paused ? <img src="play.png" alt="Resume" /> : <img src="pause.png" alt="Pause" />}
  </button>
  {/* Conditionally render another button only when video is paused */}
  {paused && (
        <button onClick={toggleDrawing}>
          <img src="draw.png" alt="Draw"/>
        </button>
      )}
  </div>
  )
  }

function Chat({socket}) {
  const [userMessage, setUserMessage] = useState('');
  const [messages, setMessages] = useState([]);
   // Set up event listener for incoming messages
   useEffect(() => {
    socket.on('message', message => {
      //console.log('Server Response:', message);
      setMessages(prevMessages => [...prevMessages, { text: message, type: 'incoming' }]);
    });

    // Clean up event listener when component unmounts
    return () => {
      socket.off('message'); // Remove the event listener
    };
  }, []); // Only run this effect when 'socket' changes
  
  const handleChat = () => {
    if (userMessage.trim() !== '') {
      
      socket.emit('outgoing', userMessage)
      setMessages(prevMessages => [...prevMessages, {text: userMessage,type: "outgoing"}]);   
      //socket.disconnect();
      setUserMessage('');
    }
  }

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault(); // Prevents the default behavior of the Enter key
      handleChat(); // Calls the handleChat function if Enter key is pressed
    }
  };

  const handleChange = (event) => {
    setUserMessage(event.target.value);
  };


  return (
     <div className= "chatArea">
      <ul className="chatBox">
        {messages.map((message, index) => (
          <li key={index} className={`chat ${message.type}`}>
            <p className="txt">{message.text}</p>
          </li>
        ))}
      </ul>
      <div className="chatFooter">
    <textarea className="txt-box" placeholder='Enter Message...' value={userMessage} onChange={handleChange}></textarea>
    <button className="send-btn" onClick={handleChat}></button>
      </div>
      </div>
  );
}

function App() { 
  const socket = io(); 
  return (
    <div className="App" style={{ backgroundColor: 'white', height: '100vh' }}>
      <div className="videoArea">
        {/* <WelcomeMessage /> */}
        <ImageStream />
      </div>
        <Chat socket = {socket}/>
      

    </div>
  );
};

export default App;

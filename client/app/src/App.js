//import NodeCount from './components/nodeCount';  // Adjust path if needed
import GenerateBanner from './components/banner'
import HomePage from './components/home';
import FedLinks from './components/links';
import AboutPage from './components/about';
import ActPage from './components/act';
import ThoughtsPage from './components/thoughts';
import {Route, Routes} from 'react-router-dom'


function App() {
  
  return (
    <div className="App">
      <GenerateBanner />
      
            
      
      <div className='container'>
        <Routes>
          <Route path='/' element = {<HomePage/>}></Route>
          <Route path='/links' element = {<FedLinks/>}></Route>
          <Route path='/about' element = {<AboutPage/>}></Route>
          <Route path='/thoughts' element = {<ThoughtsPage/>}></Route>
          <Route path='/act' element = {<ActPage/>}></Route>
        </Routes>
      </div>
      
      
    </div>
  );
}

export default App;
import express from 'express';
import driver from './bd.js';  // Adjust the path if necessary
import cors from 'cors';  // If you're using ES6 modules
import * as dotenv from 'dotenv'
import  OpenAI from "openai";
import  bodyParser from 'body-parser' ;


const PORT = 3001;
dotenv.config(); 

const openai = new OpenAI({
  organization: "org-pB15yRN3A8XpNJXGh89xUHbV",
  apiKey: process.env.OPENAI_API_KEY
});

const app = express()
app.use(cors())
app.use(express.json())
app.use(express.static('public'));
app.use(bodyParser.json());
app.get('/node-count', async (req, res) => {
  const session = driver.session();
  
  try {
      // This Cypher query gets the total number of nodes in the database
      const result = await session.run("MATCH (n) RETURN count(n) as nodeCount");
      
      // Extracting the count from the query result
      const nodeCount = result.records[0].get('nodeCount').toNumber();
      
      console.log(`Total number of nodes: ${nodeCount}`);
      
      res.send(`Total number of nodes: ${nodeCount}`);
  } catch (error) {
      console.error('Error querying the database:', error);
      res.status(500).send('Error querying the database');
  } finally {
      session.close();
  }
});


app.post('/api/signup', async (req, res) => {
  console.log("trying realy hard to add userrrr !! ")
    const { username, password } = req.body; // Extract info from request

    const session = driver.session();
    try {
      console.log("trying realy hard to add userrrr !! ")
        const result = await session.run(
            'CREATE (u:User { username: $username, password: $password , email: $email}) RETURN u',
            { username, password, email }
        );
        res.json({ success: true });
    } catch (error) {
        res.json({ success: false, message: error.message });
    } finally {
        session.close();
    }
});

app.get('/', async (req, res) => {
  res.status(200).send({
    message: 'Work under way!'
  })
})

app.post('/', async (req, res) => {
  const {prompt} = req.body;
  try {    
    const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [{ "role": "user", "content": prompt }],
        temperature: 0.35, // Higher values means the model will take more risks.
      max_tokens: 3000, // The maximum number of tokens to generate in the completion. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
      top_p: 1, // alternative to sampling with temperature, called nucleus sampling
      frequency_penalty: 0.5, // Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
      presence_penalty: 0, // Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
      });
    res.status(200).send({
        bot: response.choices[0].message
      });
   
    

/*
    const response = await openai.complete({
      model: "text-davinci-003",
      prompt: `${prompt}`,
      temperature: 0.35, // Higher values means the model will take more risks.
      max_tokens: 3000, // The maximum number of tokens to generate in the completion. Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
      top_p: 1, // alternative to sampling with temperature, called nucleus sampling
      frequency_penalty: 0.5, // Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
      presence_penalty: 0, // Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    });
*/
    

  } catch (error) {
    console.error(error)
    res.status(500).send(error || 'We have a problem.');
  }
})



app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

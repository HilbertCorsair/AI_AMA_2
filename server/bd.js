import neo4j from 'neo4j-driver';

const uri = "bolt://localhost:7687";  // default URI for local Neo4j instance
const user = "neo4j";  // default username
const password = "1848_sparrow";  // replace with the password you set

const driver = neo4j.driver(uri, neo4j.auth.basic(user, password));

export default driver;

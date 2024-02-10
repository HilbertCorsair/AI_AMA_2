// src/components/AuthenticationButtons.js

import React, { useState } from 'react';
import './signin.css'

function AuthenticationButtons(props) {
    return (
        <div id = "registration">
            <button onClick={props.onSignupClick}>Sign Up</button>
            <button onClick={props.onLoginClick}>Log In</button>
        </div>
    );
}

function LoginForm(props) {
    const [formData, setFormData] = useState({
        username: '',
        password: ''
    });

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        // Handle response as needed
    };

    return (
        <form onSubmit={handleSubmit}>
            <input type="text" name="username" value={formData.username} onChange={handleInputChange} placeholder="Username" />
            <input type="password" name="password" value={formData.password} onChange={handleInputChange} placeholder="Password" />
            <button type="submit">Log In</button>
        </form>
    );
}

function SignupForm(props) {
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        confirmPassword: '',
        email: ''// add more fields as needed
    });

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        // Send the formData to the backend using fetch or axios
        const response = await fetch('http://localhost:3001/api/signup', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        console.log(response)
    };

    return (
        <form onSubmit={handleSubmit}>
            <input type="text" name="username" value={formData.username} onChange={handleInputChange} placeholder="Username" />
            <input type="password" name="password" value={formData.password} onChange={handleInputChange} placeholder="Password" />
            <input type="confirmPassword" name="confirmPassword" value={formData.confirmPassword} onChange={handleInputChange} placeholder="Password" />
            <input type="email" name="email" value={formData.email} onChange={handleInputChange} placeholder="email" />
            {/* Add more fields as needed */}
            <button type="submit">Sign Up</button>
        </form>
    );
}

export  {SignupForm, AuthenticationButtons, LoginForm}


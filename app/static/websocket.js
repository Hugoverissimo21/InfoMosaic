document.addEventListener('DOMContentLoaded', function () {
    // Connect to the WebSocket server
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    // Listen for status updates from the backend
    socket.on('status', function(data) {
        var statusDiv = document.getElementById('status');
        statusDiv.innerHTML += '<p>' + data.message + '</p>';  // Update the status message on the page
    });

    // Handle the form submission using AJAX (prevent page refresh)
    document.getElementById('searchForm').addEventListener('submit', function(event) {
        event.preventDefault();  // Prevent the form from submitting and refreshing the page

        // Get the keyword entered by the user
        var keyword = document.getElementById('keyword').value;

        // Check if the input is empty
        if (keyword.trim() === '') {
            alert('Please enter a search term!');
            return;
        }

        // Clear any previous status messages when starting a new search
        var statusDiv = document.getElementById('status');
        statusDiv.innerHTML = '';  // Clear the status div

        // Update the URL with the query parameter
        var newUrl = '/search?keyword=' + encodeURIComponent(keyword);
        window.history.pushState({ path: newUrl }, '', newUrl);  // Update the browser's URL without reloading the page

        // Send the search term to Flask via AJAX (GET request)
        fetch(newUrl)  // Use the updated URL with the keyword
            .then(response => response.text())
            .catch(error => console.error('Error:', error));
    });
});

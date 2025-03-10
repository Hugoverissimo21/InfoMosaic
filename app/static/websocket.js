document.addEventListener('DOMContentLoaded', function () {
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    // Listen for WebSocket status updates from Flask
    socket.on('status', function(data) {
        var statusDiv = document.getElementById('status');
        statusDiv.innerHTML += '<p>' + data.message + '</p>';
    });

    // Function to handle a search query
    function handleSearch(query) {
        if (!query || query.trim() === '') {
            alert('Please enter a search term!');
            return;
        }

        var statusDiv = document.getElementById('status');
        statusDiv.innerHTML = '';  // Clear previous messages

        var newUrl = '/search?query=' + encodeURIComponent(query);
        window.history.pushState({ path: newUrl }, '', newUrl);  // Update the URL without reloading

        // Send the search term to Flask using AJAX (GET request)
        fetch(newUrl)
            .then(response => response.text())  // Expecting a complete HTML response
            .then(data => {
                // Replace the content of the page with the new response
                document.body.innerHTML = data;
                // This will trigger the page reload as needed after rendering.
            })
            .catch(error => console.error('Error:', error));
    }

    // Handle form submission
    document.getElementById('searchForm').addEventListener('submit', function(event) {
        event.preventDefault();  // Prevent page refresh
        var keyword = document.getElementById('keyword').value;
        handleSearch(keyword);
    });
});
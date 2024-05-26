pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract MuteAmplifierRedux is Ownable {
    using SafeERC20 for IERC20;
    using SafeMath for uint256;

    IERC20 public token;
    uint256 public amplifier;
    uint256 public totalSupply;
    uint256 public lastUpdateBlock;

    mapping(address => uint256) public balances;
    mapping(address => uint256) public lastRewardBlock;
    mapping(address => uint256) public pendingReward;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event RewardAdded(uint256 reward);
    event RewardPaid(address indexed user, uint256 reward);

    constructor(IERC20 _token, uint256 _amplifier) {
        token = _token;
        amplifier = _amplifier;
        lastUpdateBlock = block.number;
    }

    function deposit(uint256 _amount) external {
        balances[msg.sender] = balances[msg.sender].add(_amount);
        totalSupply = totalSupply.add(_amount);
        token.safeTransferFrom(msg.sender, address(this), _amount);
        updatePendingReward(msg.sender);
    }

    function withdraw(uint256 _amount) external {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        balances[msg.sender] = balances[msg.sender].sub(_amount);
        totalSupply = totalSupply.sub(_amount);
        token.safeTransfer(msg.sender, _amount);
        updatePendingReward(msg.sender);
    }

    function updatePendingReward(address _user) internal {
        uint256 currentBlock = block.number;
        uint256 reward = (currentBlock - lastUpdateBlock) * amplifier;
        pendingReward[_user] = pendingReward[_user].add(reward);
    }

    function payReward(address _user) external {
        require(pendingReward[_user] > 0, "No pending reward");
        uint256 reward = pendingReward[_user];
        balances[_user] = balances[_user].sub(reward);
        totalSupply = totalSupply.sub(reward);
        pendingReward[_user] = 0;
        token.safeTransfer(_user, reward);
        emit RewardPaid(_user, reward);
    }

    function addReward(uint256 _amount) external onlyOwner {
        require(_amount > 0, "Invalid reward amount");
        totalSupply = totalSupply.add(_amount);
        token.safeTransfer(address(this), _amount);
        lastUpdateBlock = block.number;
        emit RewardAdded(_amount);
    }

    function getBalance(address _user) external view returns (uint256) {
        return balances[_user];
    }

    function getPendingReward(address _user) external view returns (uint256) {
        return pendingReward[_user];
    }

    function getTotalSupply() external view returns (uint256) {
        return totalSupply;
    }

    function getLastUpdateBlock() external view returns (uint256) {
        return lastUpdateBlock;
    }

    function getAmplifier() external view returns (uint256) {
        return amplifier;
    }

    function getToken() external view returns (IERC20) {
        return token;
    }

    function approve(address _spender, uint256 _value) external returns (bool) {
        token.approve(_spender, _value);
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transfer(address _to, uint256 _value) external returns (bool) {
        token.transfer(_to, _value);
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) external returns (bool) {
        token.transferFrom(_from, _to, _value);
        emit Transfer(_from, _to, _value);
        return true;
    }
}